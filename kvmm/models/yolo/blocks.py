from keras import layers
import keras
import math


def conv_block(x, c2, k=1, s=1, g=1, d=1, act=True, data_format="channels_last", name_prefix="conv"):
    if s > 1:
        p = (k - 1) // 2
        x = layers.ZeroPadding2D(padding=(p, p), data_format=data_format,
                                name=f"{name_prefix}_pad")(x)
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
        kernel_initializer='he_normal',
        name=f"{name_prefix}.conv"
    )(x)

    axis = -1 if data_format == "channels_last" else 1
    x = layers.BatchNormalization(
        axis=axis,
        momentum=0.97,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        moving_mean_initializer='zeros',
        moving_variance_initializer='ones',
        name=f"{name_prefix}.bn"
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

    y = conv_block(x, c_, k=1, s=1, act=True, data_format=data_format, name_prefix=f"{name_prefix}.cv1")
    pool1 = layers.MaxPooling2D(
        pool_size=k,
        strides=1,
        padding='same',
        data_format=data_format,
        name=f"{name_prefix}_pool"
    )(y)

    pool2 = layers.MaxPooling2D(
        pool_size=k,
        strides=1,
        padding='same',
        data_format=data_format,
        name=f"{name_prefix}_pool_2"
    )(pool1)

    pool3 = layers.MaxPooling2D(
        pool_size=k,
        strides=1,
        padding='same',
        data_format=data_format,
        name=f"{name_prefix}_pool_3"
    )(pool2)

    if data_format == "channels_last":
        concat_axis = -1
    else:
        concat_axis = 1

    concatenated = layers.Concatenate(axis=concat_axis, name=f"{name_prefix}_concat")([y, pool1, pool2, pool3])
    output = conv_block(concatenated, c2, k=1, s=1, act=True, data_format=data_format, name_prefix=f"{name_prefix}.cv2")
    return output

def bottleneck_block(x, c2, shortcut=True, g=1, k=(1, 3), e=1.0, data_format="channels_last", name_prefix="bottleneck"):
    if data_format == "channels_last":
        c1 = x.shape[-1]
    else:
        c1 = x.shape[1]

    c_ = int(c2 * e)  # For bottlenecks in C3, e=1.0, so c_ = c2

    y = conv_block(x, c_, k=k[0], s=1, act=True, data_format=data_format, name_prefix=f"{name_prefix}.cv1")
    y = conv_block(y, c2, k=k[1], s=1, g=g, act=True, data_format=data_format, name_prefix=f"{name_prefix}.cv2")
    add_shortcut = shortcut and c1 == c2
    if add_shortcut:
        y = layers.Add(name=f"{name_prefix}_add")([x, y])

    return y


def c3_block(x, c2, n=1, shortcut=True, g=1, e=0.5, data_format="channels_last", name_prefix="c3"):
    if data_format == "channels_last":
        c1 = x.shape[-1]
    else:
        c1 = x.shape[1]
    c_ = int(c2 * e)

    branch1 = conv_block(x, c_, k=1, s=1, act=True, data_format=data_format, name_prefix=f"{name_prefix}.cv1")
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
            name_prefix=f"{name_prefix}.m.{i}"
        )

    branch2 = conv_block(x, c_, k=1, s=1, act=True, data_format=data_format, name_prefix=f"{name_prefix}.cv2")

    if data_format == "channels_last":
        concat_axis = -1
    else:
        concat_axis = 1
    concatenated = layers.Concatenate(axis=concat_axis, name=f"{name_prefix}_concat")([current, branch2])
    output = conv_block(concatenated, c2, k=1, s=1, act=True, data_format=data_format, name_prefix=f"{name_prefix}.cv3")
    return output



def detect_head(feature_maps, nc=80, reg_max=16, data_format="channels_last", name_prefix="detect"):
    nl = len(feature_maps)

    if data_format == "channels_last":
        ch = [x.shape[-1] for x in feature_maps]
    else:
        ch = [x.shape[1] for x in feature_maps]

    # Match PyTorch implementation exactly
    # cv2 (regression): c2 = 64 for all scales
    # cv3 (classification): c3 = 128 for scale 0, 128 for scale 1, 128 for scale 2
    c2 = 64  # Fixed value from PyTorch model
    c3_values = [128, 128, 128]  # Fixed values from PyTorch model

    outputs = []

    for i, x in enumerate(feature_maps):
        c3 = c3_values[i]

        # Regression branch (cv2)
        reg_branch = conv_block(x, c2, k=3, s=1, act=True, data_format=data_format,
                               name_prefix=f"{name_prefix}.cv2.{i}.0")

        reg_branch = conv_block(reg_branch, c2, k=3, s=1, act=True, data_format=data_format,
                               name_prefix=f"{name_prefix}.cv2.{i}.1")

        reg_output = layers.Conv2D(
            filters=4 * reg_max,  # 64 channels
            kernel_size=1,
            strides=1,
            padding='valid',
            use_bias=True,
            data_format=data_format,
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            name=f"{name_prefix}.cv2_new_conv.{i}.2"
        )(reg_branch)

        # Classification branch (cv3)
        cls_branch = conv_block(x, c3, k=3, s=1, act=True, data_format=data_format,
                               name_prefix=f"{name_prefix}.cv3.{i}.0")

        cls_branch = conv_block(cls_branch, c3, k=3, s=1, act=True, data_format=data_format,
                               name_prefix=f"{name_prefix}.cv3.{i}.1")

        cls_output = layers.Conv2D(
            filters=nc,  # 80 channels
            kernel_size=1,
            strides=1,
            padding='valid',
            use_bias=True,
            data_format=data_format,
            kernel_initializer='he_normal',
            bias_initializer=keras.initializers.Constant(value=-math.log((1 - 0.01) / 0.01)),
            name=f"{name_prefix}.cv3_new_conv.{i}.2"
        )(cls_branch)

        if data_format == "channels_last":
            concat_axis = -1
        else:
            concat_axis = 1

        combined_output = layers.Concatenate(axis=concat_axis, name=f"{name_prefix}_concat_{i}")([reg_output, cls_output])
        outputs.append(combined_output)

    return outputs