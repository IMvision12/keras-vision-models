from keras import layers


def conv_block(
    x, c2, k=1, s=1, g=1, d=1, act=True, data_format="channels_last", name_prefix="conv"
):
    if s > 1:
        p = (k - 1) // 2
        x = layers.ZeroPadding2D(
            padding=(p, p), data_format=data_format, name=f"{name_prefix}_pad"
        )(x)
        padding = "valid"
    else:
        padding = "same"

    x = layers.Conv2D(
        filters=c2,
        kernel_size=k,
        strides=s,
        padding=padding,
        groups=g,
        dilation_rate=d,
        use_bias=False,
        data_format=data_format,
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv",
    )(x)

    axis = -1 if data_format == "channels_last" else 1
    x = layers.BatchNormalization(
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
    )(x)

    if act is True:
        x = layers.Activation("swish", name=f"{name_prefix}_act")(x)
    elif isinstance(act, str):
        x = layers.Activation(act, name=f"{name_prefix}_act")(x)
    elif act is not False and act is not None:
        x = act(x)

    return x


def sppf_block(x, c2, k=5, data_format="channels_last", name_prefix="sppf"):
    if data_format == "channels_last":
        c1 = x.shape[-1]
    else:
        c1 = x.shape[1]

    c_ = c1 // 2

    y = conv_block(
        x,
        c_,
        k=1,
        s=1,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}.cv1",
    )
    pool1 = layers.MaxPooling2D(
        pool_size=k,
        strides=1,
        padding="same",
        data_format=data_format,
        name=f"{name_prefix}_maxpool_1",
    )(y)

    pool2 = layers.MaxPooling2D(
        pool_size=k,
        strides=1,
        padding="same",
        data_format=data_format,
        name=f"{name_prefix}_maxpool_2",
    )(pool1)

    pool3 = layers.MaxPooling2D(
        pool_size=k,
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
        k=1,
        s=1,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}_cv2",
    )
    return output


def bottleneck_block(
    x,
    c2,
    shortcut=True,
    g=1,
    k=(1, 3),
    e=1.0,
    data_format="channels_last",
    name_prefix="bottleneck",
):
    if data_format == "channels_last":
        c1 = x.shape[-1]
    else:
        c1 = x.shape[1]

    c_ = int(c2 * e)

    y = conv_block(
        x,
        c_,
        k=k[0],
        s=1,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}_cv1",
    )
    y = conv_block(
        y,
        c2,
        k=k[1],
        s=1,
        g=g,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}_cv2",
    )
    add_shortcut = shortcut and c1 == c2
    if add_shortcut:
        y = layers.Add(name=f"{name_prefix}_add")([x, y])

    return y


def c3_block(
    x, c2, n=1, shortcut=True, g=1, e=0.5, data_format="channels_last", name_prefix="c3"
):
    c_ = int(c2 * e)

    branch1 = conv_block(
        x,
        c_,
        k=1,
        s=1,
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
            g=g,
            k=(1, 3),
            e=1.0,
            data_format=data_format,
            name_prefix=f"{name_prefix}_m_{i}",
        )

    branch2 = conv_block(
        x,
        c_,
        k=1,
        s=1,
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
        k=1,
        s=1,
        act=True,
        data_format=data_format,
        name_prefix=f"{name_prefix}_cv3",
    )
    return output
