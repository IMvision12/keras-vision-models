from keras import layers, ops


def conv_block(
    inputs,
    c2,
    kernel_size=1,
    strides=1,
    groups=1,
    dilation=1,
    act=True,
    data_format="channels_last",
    name_prefix="conv",
):
    """
    Creates a convolutional block with Conv2D, BatchNormalization, and activation.

    Args:
        inputs: Input tensor
        c2: Number of output filters
        kernel_size: Size of the convolution kernel (default: 1)
        strides: Stride of the convolution (default: 1)
        groups: Number of groups for grouped convolution (default: 1)
        dilation: Dilation rate for dilated convolution (default: 1)
        act: Activation function - True for 'swish', string for specific activation,
             callable for custom activation, False/None for no activation (default: True)
        data_format: Data format, either 'channels_last' or 'channels_first' (default: 'channels_last')
        name_prefix: Prefix for layer names (default: 'conv')

    Returns:
        Output tensor after convolution, batch normalization, and activation
    """
    if strides > 1:
        p = (kernel_size - 1) // 2
        inputs = layers.ZeroPadding2D(
            padding=(p, p), data_format=data_format, name=f"{name_prefix}_pad"
        )(inputs)
        padding = "valid"
    else:
        padding = "same"

    inputs = layers.Conv2D(
        filters=c2,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        groups=groups,
        dilation_rate=dilation,
        use_bias=False,
        data_format=data_format,
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv",
    )(inputs)

    axis = -1 if data_format == "channels_last" else 1
    inputs = layers.BatchNormalization(
        axis=axis,
        momentum=0.97,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        name=f"{name_prefix}_batchnorm",
    )(inputs)

    if act is True:
        inputs = layers.Activation("swish", name=f"{name_prefix}_act")(inputs)
    elif isinstance(act, str):
        inputs = layers.Activation(act, name=f"{name_prefix}_act")(inputs)
    elif act is not False and act is not None:
        inputs = act(inputs)

    return inputs


def sppf_block(
    inputs, c2, kernel_size=5, data_format="channels_last", name_prefix="sppf"
):
    """
    Spatial Pyramid Pooling - Fast (SPPF) block.

    Applies multiple max pooling operations sequentially and concatenates the results
    to create multi-scale feature representations.

    Args:
        inputs: Input tensor
        c2: Number of output filters
        kernel_size: Size of the pooling kernel (default: 5)
        data_format: Data format, either 'channels_last' or 'channels_first' (default: 'channels_last')
        name_prefix: Prefix for layer names (default: 'sppf')

    Returns:
        Output tensor with multi-scale pooled features
    """
    if data_format == "channels_last":
        c1 = inputs.shape[-1]
    else:
        c1 = inputs.shape[1]

    c_ = c1 // 2

    y = conv_block(
        inputs,
        c_,
        kernel_size=1,
        strides=1,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}.cv1",
    )
    pool1 = layers.MaxPooling2D(
        pool_size=kernel_size,
        strides=1,
        padding="same",
        data_format=data_format,
        name=f"{name_prefix}_maxpool_1",
    )(y)

    pool2 = layers.MaxPooling2D(
        pool_size=kernel_size,
        strides=1,
        padding="same",
        data_format=data_format,
        name=f"{name_prefix}_maxpool_2",
    )(pool1)

    pool3 = layers.MaxPooling2D(
        pool_size=kernel_size,
        strides=1,
        padding="same",
        data_format=data_format,
        name=f"{name_prefix}_maxpool_3",
    )(pool2)

    if data_format == "channels_last":
        concat_axis = -1
    else:
        concat_axis = 1

    concatenated = layers.Concatenate(axis=concat_axis, name=f"{name_prefix}_concat")(
        [y, pool1, pool2, pool3]
    )
    output = conv_block(
        concatenated,
        c2,
        kernel_size=1,
        strides=1,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}_cv2",
    )
    return output


def bottleneck_block(
    inputs,
    c2,
    shortcut=True,
    groups=1,
    kernel_size=(1, 3),
    e=1.0,
    data_format="channels_last",
    name_prefix="bottleneck",
):
    """
    Bottleneck block with optional shortcut connection.

    Applies two convolutional blocks with different kernel sizes and optionally
    adds a residual connection if input and output channels match.

    Args:
        inputs: Input tensor
        c2: Number of output filters
        shortcut: Whether to add shortcut connection when possible (default: True)
        groups: Number of groups for grouped convolution (default: 1)
        kernel_size: Tuple of kernel sizes for the two conv blocks (default: (1, 3))
        e: Expansion factor for intermediate channels (default: 1.0)
        data_format: Data format, either 'channels_last' or 'channels_first' (default: 'channels_last')
        name_prefix: Prefix for layer names (default: 'bottleneck')

    Returns:
        Output tensor, optionally with residual connection
    """
    if data_format == "channels_last":
        c1 = inputs.shape[-1]
    else:
        c1 = inputs.shape[1]

    c_ = int(c2 * e)

    y = conv_block(
        inputs,
        c_,
        kernel_size=kernel_size[0],
        strides=1,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}_cv1",
    )
    y = conv_block(
        y,
        c2,
        kernel_size=kernel_size[1],
        strides=1,
        groups=groups,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}_cv2",
    )
    add_shortcut = shortcut and c1 == c2
    if add_shortcut:
        y = layers.Add(name=f"{name_prefix}_add")([inputs, y])

    return y


def c3_block(
    inputs,
    c2,
    n=1,
    shortcut=True,
    groups=1,
    e=0.5,
    data_format="channels_last",
    name_prefix="c3",
):
    """
    C3 block - Cross Stage Partial (CSP) bottleneck with 3 convolutions.

    Creates two parallel branches: one with n bottleneck blocks and one with
    a single convolution, then concatenates and processes the results.

    Args:
        inputs: Input tensor
        c2: Number of output filters
        n: Number of bottleneck blocks in the main branch (default: 1)
        shortcut: Whether to use shortcut connections in bottleneck blocks (default: True)
        groups: Number of groups for grouped convolution (default: 1)
        e: Expansion factor for intermediate channels (default: 0.5)
        data_format: Data format, either 'channels_last' or 'channels_first' (default: 'channels_last')
        name_prefix: Prefix for layer names (default: 'c3')

    Returns:
        Output tensor after CSP processing
    """
    c_ = int(c2 * e)

    branch1 = conv_block(
        inputs,
        c_,
        kernel_size=1,
        strides=1,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}_cv1",
    )
    current = branch1
    for i in range(n):
        current = bottleneck_block(
            current,
            c_,
            shortcut=shortcut,
            groups=groups,
            kernel_size=(1, 3),
            e=1.0,
            data_format=data_format,
            name_prefix=f"{name_prefix}_m_{i}",
        )

    branch2 = conv_block(
        inputs,
        c_,
        kernel_size=1,
        strides=1,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}_cv2",
    )

    if data_format == "channels_last":
        concat_axis = -1
    else:
        concat_axis = 1
    concatenated = layers.Concatenate(axis=concat_axis, name=f"{name_prefix}_concat")(
        [current, branch2]
    )
    output = conv_block(
        concatenated,
        c2,
        kernel_size=1,
        strides=1,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}_cv3",
    )
    return output


def c2f_block(
    inputs,
    c2,
    n=1,
    shortcut=False,
    groups=1,
    e=0.5,
    data_format="channels_last",
    name_prefix="c2f",
):
    """
    C2f block - Faster Implementation of CSP Bottleneck with 2 convolutions.

    This is an optimized version of the CSP bottleneck that uses fewer convolutions
    and a more efficient concatenation strategy. It splits the input into two parts,
    processes one part through n bottleneck blocks, and concatenates all intermediate
    results for better gradient flow.

    Args:
        inputs: Input tensor
        c2: Number of output filters
        n: Number of bottleneck blocks (default: 1)
        shortcut: Whether to use shortcut connections in bottleneck blocks (default: False)
        groups: Number of groups for grouped convolution (default: 1)
        e: Expansion factor for hidden channels (default: 0.5)
        data_format: Data format, either 'channels_last' or 'channels_first' (default: 'channels_last')
        name_prefix: Prefix for layer names (default: 'c2f')

    Returns:
        Output tensor after C2f processing
    """
    c_ = int(c2 * e)

    y = conv_block(
        inputs,
        2 * c_,
        kernel_size=1,
        strides=1,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}_cv1",
    )

    if data_format == "channels_last":
        split_axis = -1
        concat_axis = -1
    else:
        split_axis = 1
        concat_axis = 1

    y1, y2 = ops.split(y, 2, axis=split_axis)
    y_list = [y1, y2]

    current = y2
    for i in range(n):
        current = bottleneck_block(
            current,
            c_,
            shortcut=shortcut,
            groups=groups,
            kernel_size=(3, 3),
            e=1.0,
            data_format=data_format,
            name_prefix=f"{name_prefix}_m_{i}",
        )
        y_list.append(current)

    concatenated = layers.Concatenate(axis=concat_axis, name=f"{name_prefix}_concat")(
        y_list
    )

    output = conv_block(
        concatenated,
        c2,
        kernel_size=1,
        strides=1,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}_cv2",
    )

    return output
