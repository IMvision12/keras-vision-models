import math

import keras
from keras import layers

from .blocks import conv_block


def detect_head(
    feature_maps, nc=80, reg_max=16, data_format="channels_last", name_prefix="detect"
):
    """
    YOLO detection head for multi-scale object detection using Keras 3.

    Creates separate regression and classification branches for each feature map level
    in a YOLO-style object detection network. The detection head processes feature maps
    from different scales to predict bounding boxes and class probabilities.

    The architecture consists of:
    - Regression branch: Two 3x3 convolutions + 1x1 output layer for bbox coordinates
    - Classification branch: Two 3x3 convolutions + 1x1 output layer for class scores
    - Both branches are applied independently to each feature map scale

    Args:
        feature_maps (List[keras.KerasTensor]): List of feature map tensors from 
            different scales/levels of the network backbone. Each tensor should have 
            shape [batch_size, height, width, channels] for channels_last format or
            [batch_size, channels, height, width] for channels_first format.
        nc (int, optional): Number of classes for classification. Defaults to 80
            (COCO dataset classes).
        reg_max (int, optional): Maximum value for regression encoding, affects 
            the number of output channels in regression branch (4 * reg_max). 
            Defaults to 16.
        data_format (str, optional): Data format specification. Either 
            'channels_last' (NHWC) or 'channels_first' (NCHW). Defaults to 
            'channels_last'.
        name_prefix (str, optional): Prefix for layer names to ensure uniqueness
            in the Keras model graph. Defaults to 'detect'.

    Returns:
        List[keras.KerasTensor]: List of output tensors, one for each input feature 
            map. Each output tensor contains regression and classification predictions
            concatenated along the channel axis. The channel dimension contains:
            - First 4*reg_max channels: regression outputs (bbox coordinates)
            - Last nc channels: classification outputs (class probabilities)
            
            Output shapes:
            - channels_last: [batch_size, height, width, 4*reg_max + nc]
            - channels_first: [batch_size, 4*reg_max + nc, height, width]
    """
    if data_format == "channels_last":
        ch = [inputs.shape[-1] for inputs in feature_maps]
    else:
        ch = [inputs.shape[1] for inputs in feature_maps]

    c2 = max((16, ch[0] // 4, reg_max * 4))
    c3 = max(ch[0], min(nc, 100))

    outputs = []

    for i, inputs in enumerate(feature_maps):
        reg_branch = conv_block(
            inputs,
            c2,
            kernel_size=3,
            strides=1,
            act=True,
            data_format=data_format,
            name_prefix=f"{name_prefix}_cv2_{i}_0",
        )

        reg_branch = conv_block(
            reg_branch,
            c2,
            kernel_size=3,
            strides=1,
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
            inputs,
            c3,
            kernel_size=3,
            strides=1,
            act=True,
            data_format=data_format,
            name_prefix=f"{name_prefix}_cv3_{i}_0",
        )

        cls_branch = conv_block(
            cls_branch,
            c3,
            kernel_size=3,
            strides=1,
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
