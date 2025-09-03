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
    Standard convolutional block: Conv2D → BatchNormalization → Activation.

    Fundamental building block that applies convolution with automatic padding handling
    for stride > 1, followed by batch normalization and configurable activation.

    Args:
        inputs (keras.KerasTensor): Input tensor
        c2 (int): Number of output filters
        kernel_size (int, optional): Kernel size. Defaults to 1.
        strides (int, optional): Stride size. Auto-pads if > 1. Defaults to 1.
        groups (int, optional): Groups for grouped convolution. Defaults to 1.
        dilation (int, optional): Dilation rate. Defaults to 1.
        act (bool|str|callable, optional): Activation - True='swish', str=name,
            callable=custom, False/None=none. Defaults to True.
        data_format (str, optional): 'channels_last' or 'channels_first'. Defaults to 'channels_last'.
        name_prefix (str, optional): Layer name prefix. Defaults to 'conv'.

    Returns:
        keras.KerasTensor: Processed tensor with applied conv → batchnorm → activation
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
    Spatial Pyramid Pooling Fast (SPPF) block for multi-scale feature extraction.

    Efficiently creates multi-scale representations by applying sequential max pooling
    operations (kernel_size x 3) and concatenating results. More efficient than
    parallel SPP by reusing pooled features.

    Args:
        inputs (keras.KerasTensor): Input feature tensor
        c2 (int): Number of output channels
        kernel_size (int, optional): Pooling kernel size. Defaults to 5.
        data_format (str, optional): 'channels_last' or 'channels_first'. Defaults to 'channels_last'.
        name_prefix (str, optional): Layer name prefix. Defaults to 'sppf'.

    Returns:
        keras.KerasTensor: Multi-scale pooled features with 4x channel expansion then reduced to c2
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
    Bottleneck block with optional residual connection.

    Two-stage convolution: 1x1 expansion → kxk processing with optional residual
    shortcut when input/output channels match. Core component of ResNet-style architectures.

    Args:
        inputs (keras.KerasTensor): Input tensor
        c2 (int): Output channels
        shortcut (bool, optional): Add residual connection if channels match. Defaults to True.
        groups (int, optional): Groups for second convolution. Defaults to 1.
        kernel_size (tuple, optional): (conv1_size, conv2_size). Defaults to (1, 3).
        e (float, optional): Channel expansion factor for intermediate layer. Defaults to 1.0.
        data_format (str, optional): 'channels_last' or 'channels_first'. Defaults to 'channels_last'.
        name_prefix (str, optional): Layer name prefix. Defaults to 'bottleneck'.

    Returns:
        keras.KerasTensor: Output with optional residual connection applied
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
    Cross Stage Partial (CSP) bottleneck with 3 convolutions - C3 block.

    CSP architecture that splits processing into two branches: main branch with n bottlenecks
    and bypass branch with single conv, then concatenates and processes results.
    Improves gradient flow and reduces parameters.

    Args:
        inputs (keras.KerasTensor): Input tensor
        c2 (int): Output channels
        n (int, optional): Number of bottleneck blocks in main branch. Defaults to 1.
        shortcut (bool, optional): Enable shortcuts in bottlenecks. Defaults to True.
        groups (int, optional): Groups for bottleneck convolutions. Defaults to 1.
        e (float, optional): Channel reduction factor (0.5 = half channels). Defaults to 0.5.
        data_format (str, optional): 'channels_last' or 'channels_first'. Defaults to 'channels_last'.
        name_prefix (str, optional): Layer name prefix. Defaults to 'c3'.

    Returns:
        keras.KerasTensor: CSP-processed output combining both branches
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
    C2f block - Faster CSP implementation with 2 convolutions and gradient concatenation.

    Optimized CSP variant that splits input, processes one half through n bottlenecks
    while concatenating all intermediate features for enhanced gradient flow.
    More efficient than C3 with better feature reuse.

    Args:
        inputs (keras.KerasTensor): Input tensor
        c2 (int): Output channels
        n (int, optional): Number of bottleneck iterations. Defaults to 1.
        shortcut (bool, optional): Enable bottleneck shortcuts. Defaults to False.
        groups (int, optional): Groups for bottleneck convolutions. Defaults to 1.
        e (float, optional): Hidden channel expansion factor. Defaults to 0.5.
        data_format (str, optional): 'channels_last' or 'channels_first'. Defaults to 'channels_last'.
        name_prefix (str, optional): Layer name prefix. Defaults to 'c2f'.

    Returns:
        keras.KerasTensor: Enhanced feature representation with improved gradient flow
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
