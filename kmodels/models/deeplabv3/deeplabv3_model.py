import keras
from keras import layers, ops, utils

from kmodels.model_registry import register_model
from kmodels.weight_utils import load_weights_from_config

from .config import DEEPLABV3_MODEL_CONFIG, DEEPLABV3_WEIGHTS_CONFIG


def build_dilated_resnet_backbone(
    input_tensor,
    backbone_variant,
    include_normalization=False,
    normalization_mode="imagenet",
):
    """Build a dilated ResNet backbone for DeepLabV3.

    Constructs a ResNet-50 or ResNet-101 backbone with dilated (atrous)
    convolutions in the last two stages, matching the torchvision
    DeepLabV3 backbone configuration (output_stride=8).

    The dilation pattern follows torchvision's replace_stride_with_dilation=[False, True, True]:
    - layer1: normal (stride=1, dilation=1)
    - layer2: normal (stride=2, dilation=1)
    - layer3: stride replaced with dilation (stride=1, dilation escalates to 2)
    - layer4: stride replaced with dilation (stride=1, dilation escalates to 4)

    Layer naming mirrors torchvision's naming convention for straightforward
    weight conversion.

    Args:
        input_tensor: Input Keras tensor.
        backbone_variant: One of "ResNet50" or "ResNet101".
        include_normalization: Whether to add ImageNormalizationLayer.
        normalization_mode: Normalization mode string.

    Returns:
        Feature tensor from the last ResNet stage (C5), with spatial
        resolution 1/8 of the input (instead of the usual 1/32).
    """
    from kmodels.layers import ImageNormalizationLayer

    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    block_repeats = {
        "ResNet50": [3, 4, 6, 3],
        "ResNet101": [3, 4, 23, 3],
    }[backbone_variant]

    x = (
        ImageNormalizationLayer(mode=normalization_mode)(input_tensor)
        if include_normalization
        else input_tensor
    )

    # Stem: conv1 (7x7, stride=2) + bn1 + relu + maxpool
    x = layers.ZeroPadding2D(padding=3, data_format=data_format)(x)
    x = layers.Conv2D(
        64,
        7,
        strides=2,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="backbone_conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name="backbone_bn1",
    )(x)
    x = layers.ReLU()(x)
    x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
    x = layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="valid",
        data_format=data_format,
    )(x)

    filters_list = [64, 128, 256, 512]
    # replace_stride_with_dilation = [False, False, True, True]
    # corresponds to layer1=False, layer2=False, layer3=True, layer4=True
    dilate_stages = [False, False, True, True]
    current_dilation = 1

    for stage_idx, num_blocks in enumerate(block_repeats):
        filters = filters_list[stage_idx]
        original_stride = 2 if stage_idx > 0 else 1

        if dilate_stages[stage_idx] and stage_idx > 0:
            current_dilation *= original_stride
            stage_stride = 1
        else:
            stage_stride = original_stride

        previous_dilation = current_dilation // (
            original_stride if dilate_stages[stage_idx] and stage_idx > 0 else 1
        )

        for block_idx in range(num_blocks):
            prefix = f"backbone_layer{stage_idx + 1}_{block_idx}"

            if block_idx == 0:
                block_stride = stage_stride
                block_dilation = previous_dilation
            else:
                block_stride = 1
                block_dilation = current_dilation

            residual = x

            # Conv 1x1 reduce
            x = layers.Conv2D(
                filters,
                1,
                strides=1,
                padding="valid",
                use_bias=False,
                data_format=data_format,
                name=f"{prefix}_conv1",
            )(x)
            x = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                momentum=0.1,
                name=f"{prefix}_bn1",
            )(x)
            x = layers.ReLU()(x)

            # Conv 3x3 with dilation
            if block_stride > 1:
                pad_size = block_dilation
                x = layers.ZeroPadding2D(padding=pad_size, data_format=data_format)(x)
                x = layers.Conv2D(
                    filters,
                    3,
                    strides=block_stride,
                    padding="valid",
                    dilation_rate=block_dilation,
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_conv2",
                )(x)
            else:
                if block_dilation > 1:
                    pad_size = block_dilation
                    x = layers.ZeroPadding2D(padding=pad_size, data_format=data_format)(
                        x
                    )
                    x = layers.Conv2D(
                        filters,
                        3,
                        strides=1,
                        padding="valid",
                        dilation_rate=block_dilation,
                        use_bias=False,
                        data_format=data_format,
                        name=f"{prefix}_conv2",
                    )(x)
                else:
                    x = layers.Conv2D(
                        filters,
                        3,
                        strides=1,
                        padding="same",
                        use_bias=False,
                        data_format=data_format,
                        name=f"{prefix}_conv2",
                    )(x)

            x = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                momentum=0.1,
                name=f"{prefix}_bn2",
            )(x)
            x = layers.ReLU()(x)

            # Conv 1x1 expand
            x = layers.Conv2D(
                filters * 4,
                1,
                strides=1,
                padding="valid",
                use_bias=False,
                data_format=data_format,
                name=f"{prefix}_conv3",
            )(x)
            x = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                momentum=0.1,
                name=f"{prefix}_bn3",
            )(x)

            # Downsample residual if needed
            in_channels = residual.shape[channels_axis]
            out_channels = filters * 4
            if block_stride != 1 or in_channels != out_channels:
                if block_stride > 1:
                    residual = layers.ZeroPadding2D(padding=0, data_format=data_format)(
                        residual
                    )
                residual = layers.Conv2D(
                    out_channels,
                    1,
                    strides=block_stride,
                    padding="valid",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_downsample_conv",
                )(residual)
                residual = layers.BatchNormalization(
                    axis=channels_axis,
                    epsilon=1e-5,
                    momentum=0.1,
                    name=f"{prefix}_downsample_bn",
                )(residual)

            x = layers.Add()([x, residual])
            x = layers.ReLU()(x)

    return x


def aspp_module(x, name="aspp"):
    """Atrous Spatial Pyramid Pooling (ASPP) module.

    Applies parallel atrous convolutions at multiple dilation rates to capture
    multi-scale context information, following the DeepLabV3 architecture.

    Branches:
    - 1x1 convolution (no dilation)
    - 3x3 convolution with dilation rate 12
    - 3x3 convolution with dilation rate 24
    - 3x3 convolution with dilation rate 36
    - Image-level pooling (global average pooling + 1x1 conv)

    The outputs are concatenated and projected to 256 channels.

    Args:
        x: Input feature tensor from the backbone (2048 channels).
        name: Name prefix for layers.

    Returns:
        Output tensor with 256 channels.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    branches = []

    # Branch 0: 1x1 convolution
    b0 = layers.Conv2D(
        256,
        1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_convs_0_0",
    )(x)
    b0 = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name=f"{name}_convs_0_1",
    )(b0)
    b0 = layers.ReLU()(b0)
    branches.append(b0)

    # Branches 1-3: 3x3 convolutions with dilation rates 12, 24, 36
    atrous_rates = [12, 24, 36]
    for i, rate in enumerate(atrous_rates, start=1):
        b = layers.Conv2D(
            256,
            3,
            padding="same",
            dilation_rate=rate,
            use_bias=False,
            data_format=data_format,
            name=f"{name}_convs_{i}_0",
        )(x)
        b = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-5,
            momentum=0.1,
            name=f"{name}_convs_{i}_1",
        )(b)
        b = layers.ReLU()(b)
        branches.append(b)

    # Branch 4: Image-level pooling
    # AdaptiveAvgPool2d(1) -> Conv2d(1x1) -> BN -> ReLU -> upsample
    input_shape = ops.shape(x)
    if data_format == "channels_last":
        target_h = input_shape[1]
        target_w = input_shape[2]
    else:
        target_h = input_shape[2]
        target_w = input_shape[3]

    b4 = layers.GlobalAveragePooling2D(
        data_format=data_format,
        keepdims=True,
        name=f"{name}_convs_4_0",
    )(x)
    b4 = layers.Conv2D(
        256,
        1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_convs_4_1",
    )(b4)
    b4 = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name=f"{name}_convs_4_2",
    )(b4)
    b4 = layers.ReLU()(b4)
    b4 = layers.Resizing(
        height=target_h,
        width=target_w,
        interpolation="bilinear",
        data_format=data_format,
        name=f"{name}_convs_4_upsample",
    )(b4)
    branches.append(b4)

    # Concatenate all branches
    x = layers.Concatenate(axis=channels_axis, name=f"{name}_concat")(branches)

    # Project: 1x1 conv (1280 -> 256) + BN + ReLU + Dropout
    x = layers.Conv2D(
        256,
        1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_project_0",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name=f"{name}_project_1",
    )(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)

    return x


def classifier_head(x, num_classes, name="classifier"):
    """DeepLabV3 classifier head.

    Applies a 3x3 convolution + BN + ReLU followed by a 1x1 classification
    convolution, matching the torchvision DeepLabHead structure (minus the ASPP
    which is handled separately).

    Args:
        x: Input tensor from ASPP (256 channels).
        num_classes: Number of output segmentation classes.
        name: Name prefix for layers.

    Returns:
        Output tensor with num_classes channels.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    x = layers.Conv2D(
        256,
        3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name=f"{name}_2",
    )(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        num_classes,
        1,
        data_format=data_format,
        name=f"{name}_4",
    )(x)

    return x


@keras.saving.register_keras_serializable(package="kmodels")
class DeepLabV3(keras.Model):
    """DeepLabV3 model for semantic segmentation.

    DeepLabV3 is a semantic segmentation model that uses a dilated (atrous)
    ResNet backbone combined with Atrous Spatial Pyramid Pooling (ASPP) to
    capture multi-scale context. This implementation follows the torchvision
    architecture.

    Architecture:
        1. A dilated ResNet backbone (ResNet-50 or ResNet-101) with output stride 8
        2. ASPP module with parallel atrous convolutions at rates 12, 24, 36
        3. A classifier head producing per-pixel class predictions
        4. Bilinear upsampling to input resolution

    Reference:
        - [Rethinking Atrous Convolution for Semantic Image Segmentation]
          (https://arxiv.org/abs/1706.05587) (Chen et al., 2017)

    Args:
        backbone_variant: ResNet variant ("ResNet50" or "ResNet101").
        num_classes: Number of output segmentation classes.
        input_shape: Input shape as (height, width, channels).
        input_tensor: Optional input tensor.
        include_normalization: Whether to add ImageNet normalization.
        normalization_mode: Normalization mode for ImageNormalizationLayer.
        name: Name for the model.
        **kwargs: Additional arguments passed to keras.Model.

    Example:
        ```python
        model = DeepLabV3(
            backbone_variant="ResNet50",
            num_classes=21,
            input_shape=(520, 520, 3),
        )
        ```
    """

    def __init__(
        self,
        backbone_variant,
        num_classes,
        input_shape=None,
        input_tensor=None,
        include_normalization=False,
        normalization_mode="imagenet",
        name="DeepLabV3",
        **kwargs,
    ):
        if input_tensor is not None:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        else:
            img_input = layers.Input(shape=input_shape)

        # Build dilated ResNet backbone
        backbone_features = build_dilated_resnet_backbone(
            img_input,
            backbone_variant,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
        )

        # ASPP
        x = aspp_module(backbone_features, name="classifier_0")

        # Classifier head
        x = classifier_head(x, num_classes, name="classifier")

        # Upsample to input resolution
        x = layers.Resizing(
            height=input_shape[0],
            width=input_shape[1],
            interpolation="bilinear",
            name=f"{name}_final_upsampling",
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.backbone_variant = backbone_variant
        self.num_classes = num_classes
        self._input_shape = input_shape
        self.input_tensor = input_tensor
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_variant": self.backbone_variant,
                "num_classes": self.num_classes,
                "input_shape": self._input_shape,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_deeplabv3_model(
    variant,
    num_classes=None,
    input_shape=None,
    input_tensor=None,
    weights=None,
    include_normalization=False,
    normalization_mode="imagenet",
    **kwargs,
):
    """Creates a DeepLabV3 model with the specified variant and configuration.

    Args:
        variant: The DeepLabV3 variant (e.g., "DeepLabV3ResNet50").
        num_classes: Number of output segmentation classes.
            If None and using pretrained weights, defaults to 21 (Pascal VOC).
        input_shape: Input shape as (height, width, channels).
            If None, defaults to (520, 520, 3).
        input_tensor: Optional input tensor.
        weights: Pretrained weights to load. Options:
            - "coco_voc": COCO-pretrained with VOC labels (21 classes)
            - None: No pretrained weights
            - Path to weights file
        include_normalization: Whether to add ImageNet normalization.
        normalization_mode: Normalization mode string.
        **kwargs: Additional arguments.

    Returns:
        Configured DeepLabV3 model.
    """
    config = DEEPLABV3_MODEL_CONFIG[variant]

    valid_model_weights = []
    if variant in DEEPLABV3_WEIGHTS_CONFIG:
        valid_model_weights = list(DEEPLABV3_WEIGHTS_CONFIG[variant].keys())

    valid_weights = [None] + valid_model_weights

    if weights not in valid_weights and not isinstance(weights, str):
        raise ValueError(
            f"Invalid weights: {weights}. "
            f"Supported weights for {variant} are "
            f"{', '.join([str(w) for w in valid_weights])}, "
            "a path to a weights file, or None."
        )

    if num_classes is None:
        if weights in valid_model_weights:
            num_classes = config["num_classes"]
            print(
                f"No num_classes specified. Using default {num_classes} (Pascal VOC)."
            )
        else:
            raise ValueError(
                "num_classes must be specified when not using pretrained weights."
            )

    if input_shape is None:
        input_shape = (520, 520, 3)
        print(f"Using default input shape {input_shape}.")

    original_num_classes = config["num_classes"]
    use_original_classes = (
        weights in valid_model_weights and num_classes != original_num_classes
    )
    model_num_classes = original_num_classes if use_original_classes else num_classes

    model = DeepLabV3(
        backbone_variant=config["backbone_variant"],
        num_classes=model_num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        name=variant,
        **kwargs,
    )

    if weights in valid_model_weights:
        print(f"Loading {weights} weights for {variant}.")
        load_weights_from_config(variant, weights, model, DEEPLABV3_WEIGHTS_CONFIG)
    elif weights is not None and isinstance(weights, str):
        print(f"Loading weights from file: {weights}")
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    if use_original_classes:
        print(
            f"Modifying classifier from {original_num_classes} to {num_classes} classes."
        )
        new_model = DeepLabV3(
            backbone_variant=config["backbone_variant"],
            num_classes=num_classes,
            input_shape=input_shape,
            input_tensor=input_tensor,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            name=variant,
            **kwargs,
        )

        # Transfer all weights except the final classifier layer
        for old_layer, new_layer in zip(model.layers, new_model.layers):
            if old_layer.name == new_layer.name and old_layer.name != "classifier_4":
                if old_layer.get_weights():
                    new_layer.set_weights(old_layer.get_weights())

        return new_model

    return model


@register_model
def DeepLabV3ResNet50(
    num_classes=None,
    input_shape=None,
    input_tensor=None,
    weights=None,
    include_normalization=False,
    normalization_mode="imagenet",
    **kwargs,
):
    return _create_deeplabv3_model(
        "DeepLabV3ResNet50",
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        **kwargs,
    )


@register_model
def DeepLabV3ResNet101(
    num_classes=None,
    input_shape=None,
    input_tensor=None,
    weights=None,
    include_normalization=False,
    normalization_mode="imagenet",
    **kwargs,
):
    return _create_deeplabv3_model(
        "DeepLabV3ResNet101",
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        **kwargs,
    )
