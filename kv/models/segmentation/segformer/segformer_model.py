import keras
from keras import layers, utils

from kv.models import MiT_B0, MiT_B1, MiT_B2, MiT_B3, MiT_B4, MiT_B5
from kv.utils import register_model
from .config import SEGFORMER_MODEL_CONFIG

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


@keras.saving.register_keras_serializable(package="kv")
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
            if input_shape is None:
                input_shape = backbone.input_shape[1:]
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
        config = {
            "backbone": self.backbone,
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "dropout_rate": self.dropout_rate,
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "name": self.name,
            "trainable": self.trainable,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_model
def SegFormerB0(
    backbone=None,
    num_classes=19,
    input_shape=(224, 224, 3),
    input_tensor=None,
    weights=None,
    backbone_weights=False,
    backbone_weight_type=None,
    **kwargs,
):
    if weights is not None and backbone_weights:
        raise ValueError(
            "Cannot use both SegFormer weights and backbone weights. "
            "Please choose either weights='ade20k'/'cityscapes' or backbone_weights=True"
        )

    if backbone is None:
        backbone = MiT_B0(
            include_top=False,
            as_backbone=True,
            input_shape=input_shape,
            weights=backbone_weight_type,
            include_preprocessing=False,
        )

    return SegFormer(
        **SEGFORMER_MODEL_CONFIG["SegFormer_B0"],
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        name="SegFormer_B0",
        **kwargs,
    )


@register_model
def SegFormerB1(
    backbone=None,
    num_classes=19,
    input_shape=(224, 224, 3),
    input_tensor=None,
    weights=None,
    backbone_weights=False,
    backbone_weight_type=None,
    **kwargs,
):
    if weights is not None and backbone_weights:
        raise ValueError(
            "Cannot use both SegFormer weights and backbone weights. "
            "Please choose either weights='ade20k'/'cityscapes' or backbone_weights=True"
        )

    if backbone is None:
        backbone = MiT_B1(
            include_top=False,
            as_backbone=True,
            input_shape=input_shape,
            weights=backbone_weight_type,
            include_preprocessing=False,
        )

    return SegFormer(
        **SEGFORMER_MODEL_CONFIG["SegFormer_B1"],
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        name="SegFormer_B1",
        **kwargs,
    )


@register_model
def SegFormerB2(
    backbone=None,
    num_classes=19,
    input_shape=(224, 224, 3),
    input_tensor=None,
    weights=None,
    backbone_weights=False,
    backbone_weight_type=None,
    **kwargs,
):
    if weights is not None and backbone_weights:
        raise ValueError(
            "Cannot use both SegFormer weights and backbone weights. "
            "Please choose either weights='ade20k'/'cityscapes' or backbone_weights=True"
        )

    if backbone is None:
        backbone = MiT_B2(
            include_top=False,
            as_backbone=True,
            input_shape=input_shape,
            weights=backbone_weight_type,
            include_preprocessing=False,
        )

    return SegFormer(
        **SEGFORMER_MODEL_CONFIG["SegFormer_B2"],
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        name="SegFormer_B2",
        **kwargs,
    )


@register_model
def SegFormerB3(
    backbone=None,
    num_classes=19,
    input_shape=(224, 224, 3),
    input_tensor=None,
    weights=None,
    backbone_weights=False,
    backbone_weight_type=None,
    **kwargs,
):
    if weights is not None and backbone_weights:
        raise ValueError(
            "Cannot use both SegFormer weights and backbone weights. "
            "Please choose either weights='ade20k'/'cityscapes' or backbone_weights=True"
        )

    if backbone is None:
        backbone = MiT_B3(
            include_top=False,
            as_backbone=True,
            input_shape=input_shape,
            weights=backbone_weight_type,
            include_preprocessing=False,
        )

    return SegFormer(
        **SEGFORMER_MODEL_CONFIG["SegFormer_B3"],
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        name="SegFormer_B3",
        **kwargs,
    )


@register_model
def SegFormerB4(
    backbone=None,
    num_classes=19,
    input_shape=(224, 224, 3),
    input_tensor=None,
    weights=None,
    backbone_weights=False,
    backbone_weight_type=None,
    **kwargs,
):
    if weights is not None and backbone_weights:
        raise ValueError(
            "Cannot use both SegFormer weights and backbone weights. "
            "Please choose either weights='ade20k'/'cityscapes' or backbone_weights=True"
        )

    if backbone is None:
        backbone = MiT_B4(
            include_top=False,
            as_backbone=True,
            input_shape=input_shape,
            weights=backbone_weight_type,
            include_preprocessing=False,
        )

    return SegFormer(
        **SEGFORMER_MODEL_CONFIG["SegFormer_B4"],
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        name="SegFormer_B4",
        **kwargs,
    )


@register_model
def SegFormerB5(
    backbone=None,
    num_classes=19,
    input_shape=(224, 224, 3),
    input_tensor=None,
    weights=None,
    backbone_weights=False,
    backbone_weight_type=None,
    **kwargs,
):
    if weights is not None and backbone_weights:
        raise ValueError(
            "Cannot use both SegFormer weights and backbone weights. "
            "Please choose either weights='ade20k'/'cityscapes' or backbone_weights=True"
        )

    if backbone is None:
        backbone = MiT_B5(
            include_top=False,
            as_backbone=True,
            input_shape=input_shape,
            weights=backbone_weight_type,
            include_preprocessing=False,
        )

    return SegFormer(
        **SEGFORMER_MODEL_CONFIG["SegFormer_B5"],
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        name="SegFormer_B5",
        **kwargs,
    )
