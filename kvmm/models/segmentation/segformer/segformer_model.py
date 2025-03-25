import keras
from keras import layers, utils

from kvmm.models import mit
from kvmm.utils import get_all_weight_names, load_weights_from_config, register_model

from .config import SEGFORMER_MODEL_CONFIG, SEGFORMER_WEIGHTS_CONFIG


def segformer_head(
    features, embed_dim=256, num_classes=19, dropout_rate=0.1, name="segformer_head"
):
    """
    Creates a SegFormer decoder head using functional API.

    Args:
        features: List of feature tensors from the backbone
        embed_dim: Embedding dimension for the linear projections
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        name: Name prefix for the layers

    Returns:
        Tensor: Output segmentation map
    """
    target_height = features[0].shape[1]
    target_width = features[0].shape[2]

    projected_features = []
    for i, feature in enumerate(features):
        x = layers.Dense(embed_dim, name=f"{name}_linear_c{i + 1}")(feature)

        x = layers.Resizing(
            height=target_height,
            width=target_width,
            interpolation="bilinear",
            name=f"{name}_resize_c{i + 1}",
        )(x)
        projected_features.append(x)

    x = layers.Concatenate(axis=-1, name=f"{name}_concat")(projected_features[::-1])

    x = layers.Conv2D(
        filters=embed_dim, kernel_size=1, use_bias=False, name=f"{name}_fusion_conv"
    )(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f"{name}_fusion_bn")(
        x
    )
    x = layers.Activation("relu", name=f"{name}_fusion_relu")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_dropout")(x)

    x = layers.Conv2D(filters=num_classes, kernel_size=1, name=f"{name}_classifier")(x)

    return x


@keras.saving.register_keras_serializable(package="kvmm")
class SegFormer(keras.Model):
    def __init__(
        self,
        backbone,
        num_classes,
        embed_dim=256,
        dropout_rate=0.1,
        input_shape=None,
        input_tensor=None,
        name="SegFormer",
        **kwargs,
    ):
        if not getattr(backbone, "as_backbone", False):
            raise ValueError(
                "The provided backbone must be initialized with as_backbone=True"
            )

        if input_tensor is not None:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        else:
            img_input = layers.Input(shape=input_shape)

        inputs = img_input

        features = backbone(inputs)

        x = segformer_head(
            features=features,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            name=f"{name}_head",
        )

        x = layers.Resizing(
            height=input_shape[0],
            width=input_shape[1],
            interpolation="bilinear",
            name=f"{name}_final_upsampling",
        )(x)

        outputs = layers.Activation("softmax", name=f"{name}_output_activation")(x)

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.backbone = backbone
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone": self.backbone,
                "num_classes": self.num_classes,
                "embed_dim": self.embed_dim,
                "dropout_rate": self.dropout_rate,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_model
def SegFormerB0(
    backbone=None,
    num_classes=None,
    input_shape=(512, 512, 3),
    input_tensor=None,
    weights="mit",
    **kwargs,
):
    if weights not in [None, "cityscapes", "ade20k", "mit"]:
        raise ValueError(
            f"Invalid weights: {weights}. "
            "Supported weights are 'cityscapes', 'ade20k', 'mit', or None."
        )

    if weights == "cityscapes" and num_classes != 19:
        raise ValueError(
            f"Invalid number of classes: {num_classes}. "
            "When using 'cityscapes' weights, num_classes must be 19."
        )

    if backbone is None:
        if weights == "mit":
            print(
                "No backbone specified. "
                "Using MiT_B0 backbone with ImageNet-1K (in1k) weights by default."
            )
            backbone_weights = "in1k"
        elif weights is None:
            print(
                "No backbone specified and no weights provided. "
                "Using MiT_B0 backbone with no pre-trained weights."
            )
            backbone_weights = None
        else:
            backbone_weights = None

        backbone = mit.MiT_B0(
            include_top=False,
            as_backbone=True,
            input_shape=input_shape,
            weights=backbone_weights,
            include_normalization=False,
        )

    model = SegFormer(
        **SEGFORMER_MODEL_CONFIG["SegFormerB0"],
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name="SegFormerB0",
        **kwargs,
    )

    if weights in get_all_weight_names(SEGFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SegFormerB0", weights, model, SEGFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SegFormerB1(
    backbone=None,
    num_classes=None,
    input_shape=(512, 512, 3),
    input_tensor=None,
    weights="mit",
    **kwargs,
):
    if weights not in [None, "cityscapes", "ade20k", "mit"]:
        raise ValueError(
            f"Invalid weights: {weights}. "
            "Supported weights are 'cityscapes', 'ade20k', 'mit', or None."
        )

    if weights == "cityscapes" and num_classes != 19:
        raise ValueError(
            f"Invalid number of classes: {num_classes}. "
            "When using 'cityscapes' weights, num_classes must be 19."
        )

    if backbone is None:
        if weights == "mit":
            print(
                "No backbone specified. "
                "Using MiT_B1 backbone with ImageNet-1K (in1k) weights by default."
            )
            backbone_weights = "in1k"
        elif weights is None:
            print(
                "No backbone specified and no weights provided. "
                "Using MiT_B1 backbone with no pre-trained weights."
            )
            backbone_weights = None
        else:
            backbone_weights = None

        backbone = mit.MiT_B1(
            include_top=False,
            as_backbone=True,
            input_shape=input_shape,
            weights=backbone_weights,
            include_normalization=False,
        )

    model = SegFormer(
        **SEGFORMER_MODEL_CONFIG["SegFormerB1"],
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name="SegFormerB1",
        **kwargs,
    )

    if weights in get_all_weight_names(SEGFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SegFormerB1", weights, model, SEGFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SegFormerB2(
    backbone=None,
    num_classes=None,
    input_shape=(512, 512, 3),
    input_tensor=None,
    weights="mit",
    **kwargs,
):
    if weights not in [None, "cityscapes", "ade20k", "mit"]:
        raise ValueError(
            f"Invalid weights: {weights}. "
            "Supported weights are 'cityscapes', 'ade20k', 'mit', or None."
        )

    if weights == "cityscapes" and num_classes != 19:
        raise ValueError(
            f"Invalid number of classes: {num_classes}. "
            "When using 'cityscapes' weights, num_classes must be 19."
        )

    if backbone is None:
        if weights == "mit":
            print(
                "No backbone specified. "
                "Using MiT_B2 backbone with ImageNet-1K (in1k) weights by default."
            )
            backbone_weights = "in1k"
        elif weights is None:
            print(
                "No backbone specified and no weights provided. "
                "Using MiT_B2 backbone with no pre-trained weights."
            )
            backbone_weights = None
        else:
            backbone_weights = None

        backbone = mit.MiT_B2(
            include_top=False,
            as_backbone=True,
            input_shape=input_shape,
            weights=backbone_weights,
            include_normalization=False,
        )

    model = SegFormer(
        **SEGFORMER_MODEL_CONFIG["SegFormerB2"],
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name="SegFormerB2",
        **kwargs,
    )

    if weights in get_all_weight_names(SEGFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SegFormerB2", weights, model, SEGFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SegFormerB3(
    backbone=None,
    num_classes=None,
    input_shape=(512, 512, 3),
    input_tensor=None,
    weights="mit",
    **kwargs,
):
    if weights not in [None, "cityscapes", "ade20k", "mit"]:
        raise ValueError(
            f"Invalid weights: {weights}. "
            "Supported weights are 'cityscapes', 'ade20k', 'mit', or None."
        )

    if weights == "cityscapes" and num_classes != 19:
        raise ValueError(
            f"Invalid number of classes: {num_classes}. "
            "When using 'cityscapes' weights, num_classes must be 19."
        )

    if backbone is None:
        if weights == "mit":
            print(
                "No backbone specified. "
                "Using MiT_B3 backbone with ImageNet-1K (in1k) weights by default."
            )
            backbone_weights = "in1k"
        elif weights is None:
            print(
                "No backbone specified and no weights provided. "
                "Using MiT_B3 backbone with no pre-trained weights."
            )
            backbone_weights = None
        else:
            backbone_weights = None

        backbone = mit.MiT_B3(
            include_top=False,
            as_backbone=True,
            input_shape=input_shape,
            weights=backbone_weights,
            include_normalization=False,
        )

    model = SegFormer(
        **SEGFORMER_MODEL_CONFIG["SegFormerB3"],
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name="SegFormerB3",
        **kwargs,
    )

    if weights in get_all_weight_names(SEGFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SegFormerB3", weights, model, SEGFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SegFormerB4(
    backbone=None,
    num_classes=None,
    input_shape=(512, 512, 3),
    input_tensor=None,
    weights="mit",
    **kwargs,
):
    if weights not in [None, "cityscapes", "ade20k", "mit"]:
        raise ValueError(
            f"Invalid weights: {weights}. "
            "Supported weights are 'cityscapes', 'ade20k', 'mit', or None."
        )

    if weights == "cityscapes" and num_classes != 19:
        raise ValueError(
            f"Invalid number of classes: {num_classes}. "
            "When using 'cityscapes' weights, num_classes must be 19."
        )

    if backbone is None:
        if weights == "mit":
            print(
                "No backbone specified. "
                "Using MiT_B4 backbone with ImageNet-1K (in1k) weights by default."
            )
            backbone_weights = "in1k"
        elif weights is None:
            print(
                "No backbone specified and no weights provided. "
                "Using MiT_B4 backbone with no pre-trained weights."
            )
            backbone_weights = None
        else:
            backbone_weights = None

        backbone = mit.MiT_B4(
            include_top=False,
            as_backbone=True,
            input_shape=input_shape,
            weights=backbone_weights,
            include_normalization=False,
        )

    model = SegFormer(
        **SEGFORMER_MODEL_CONFIG["SegFormerB4"],
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name="SegFormerB4",
        **kwargs,
    )

    if weights in get_all_weight_names(SEGFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SegFormerB4", weights, model, SEGFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SegFormerB5(
    backbone=None,
    num_classes=1000,
    input_shape=(512, 512, 3),
    input_tensor=None,
    weights="mit",
    **kwargs,
):
    if weights not in [None, "cityscapes", "ade20k", "mit"]:
        raise ValueError(
            f"Invalid weights: {weights}. "
            "Supported weights are 'cityscapes', 'ade20k', 'mit', or None."
        )

    if weights == "cityscapes" and num_classes != 19:
        raise ValueError(
            f"Invalid number of classes: {num_classes}. "
            "When using 'cityscapes' weights, num_classes must be 19."
        )

    if backbone is None:
        if weights == "mit":
            print(
                "No backbone specified. "
                "Using MiT_B5 backbone with ImageNet-1K (in1k) weights by default."
            )
            backbone_weights = "in1k"
        elif weights is None:
            print(
                "No backbone specified and no weights provided. "
                "Using MiT_B5 backbone with no pre-trained weights."
            )
            backbone_weights = None
        else:
            backbone_weights = None

        backbone = mit.MiT_B5(
            include_top=False,
            as_backbone=True,
            input_shape=input_shape,
            weights=backbone_weights,
            include_normalization=False,
        )

    model = SegFormer(
        **SEGFORMER_MODEL_CONFIG["SegFormerB5"],
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name="SegFormerB5",
        **kwargs,
    )

    if weights in get_all_weight_names(SEGFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SegFormerB5", weights, model, SEGFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
