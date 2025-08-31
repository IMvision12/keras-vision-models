import math

import keras
from keras import layers

from .blocks import conv_block


def detect_head(
    feature_maps, nc=80, reg_max=16, data_format="channels_last", name_prefix="detect"
):
    if data_format == "channels_last":
        ch = [x.shape[-1] for x in feature_maps]
    else:
        ch = [x.shape[1] for x in feature_maps]

    c2 = max((16, ch[0] // 4, reg_max * 4))
    c3 = max(ch[0], min(nc, 100))

    outputs = []

    for i, x in enumerate(feature_maps):
        reg_branch = conv_block(
            x,
            c2,
            k=3,
            s=1,
            act=True,
            data_format=data_format,
            name_prefix=f"{name_prefix}_cv2_{i}_0",
        )

        reg_branch = conv_block(
            reg_branch,
            c2,
            k=3,
            s=1,
            act=True,
            data_format=data_format,
            name_prefix=f"{name_prefix}_cv2_{i}_1",
        )

        reg_output = layers.Conv2D(
            filters=4 * reg_max,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=True,
            data_format=data_format,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            name=f"{name_prefix}_cv2_head_conv_{i}_2",
        )(reg_branch)

        cls_branch = conv_block(
            x,
            c3,
            k=3,
            s=1,
            act=True,
            data_format=data_format,
            name_prefix=f"{name_prefix}_cv3_{i}_0",
        )

        cls_branch = conv_block(
            cls_branch,
            c3,
            k=3,
            s=1,
            act=True,
            data_format=data_format,
            name_prefix=f"{name_prefix}_cv3_{i}_1",
        )

        cls_output = layers.Conv2D(
            filters=nc,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=True,
            data_format=data_format,
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(
                value=-math.log((1 - 0.01) / 0.01)
            ),
            name=f"{name_prefix}_cv3_head_conv_{i}_2",
        )(cls_branch)

        if data_format == "channels_last":
            concat_axis = -1
        else:
            concat_axis = 1

        combined_output = layers.Concatenate(
            axis=concat_axis, name=f"{name_prefix}_concat_{i}"
        )([reg_output, cls_output])
        outputs.append(combined_output)

    return outputs
