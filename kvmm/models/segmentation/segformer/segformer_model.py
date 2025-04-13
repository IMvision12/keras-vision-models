import keras
from keras import layers, utils

from kvmm.model_registry import register_model
from kvmm.models import mit
from kvmm.utils import load_weights_from_config

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
    """
    SegFormer model for semantic segmentation tasks.

    SegFormer is a semantic segmentation architecture that combines a hierarchical
    Transformer-based encoder (MiT) with a lightweight all-MLP decoder. This class
    implements the complete SegFormer model as described in the paper:
    "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
    (Xie et al., 2021).

    The model consists of:
    1. A backbone network (typically MiT) that extracts multi-scale features
    2. A lightweight all-MLP decoder that aggregates the multi-scale features
    3. A segmentation head that produces the final pixel-wise class predictions

    Args:
        backbone (keras.Model): A backbone model that outputs a list of feature maps
            at different scales. The backbone must be initialized with `as_backbone=True`.

        num_classes (int): Number of output classes for segmentation.

        embed_dim (int, optional): Embedding dimension for the MLP decoder.
            Default: 256

        dropout_rate (float, optional): Dropout rate applied before the final
            classification layer. Must be between 0 and 1.
            Default: 0.1

        input_shape (tuple, optional): The input shape in the format (height, width, channels).
            Only used if `input_tensor` is not provided.
            Default: None

        input_tensor (Tensor, optional): Optional input tensor to use instead of creating
            a new input layer. This is useful when connecting this model as part of a
            larger model.
            Default: None

        name (str, optional): Name for the model.
            Default: "SegFormer"

        **kwargs: Additional keyword arguments passed to the keras.Model parent class.

    Returns:
        A Keras model instance with the SegFormer architecture.

    Example:
        ```python
        # Create a MiT backbone
        backbone = mit.MiT_B0(
            include_top=False,
            input_shape=(512, 512, 3),
            as_backbone=True,
        )

        # Create a SegFormer model with the backbone
        model = SegFormer(
            backbone=backbone,
            num_classes=19,
            embed_dim=256,
        )
        ```

    Note:
        The backbone is expected to return a list of feature tensors at different
        scales. The SegFormer architecture is specifically designed to work well
        with the Mix Transformer (MiT) backbone, but can be used with other
        backbones that return similar multi-scale features.
    """

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

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.backbone = backbone
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        backbone_config = keras.saving.serialize_keras_object(self.backbone)
        config.update(
            {
                "backbone": backbone_config,
                "num_classes": self.num_classes,
                "embed_dim": self.embed_dim,
                "dropout_rate": self.dropout_rate,
                "input_shape": self.input_shape[1:],
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if isinstance(config["backbone"], dict):
            config["backbone"] = keras.saving.deserialize_keras_object(
                config["backbone"]
            )
        return cls(**config)


def _create_segformer_model(
    variant,
    backbone=None,
    num_classes=None,
    input_shape=None,
    input_tensor=None,
    weights="mit",
    **kwargs,
):
    """
    Creates a SegFormer model with the specified variant and configuration.

    This helper function handles the creation of SegFormer semantic segmentation models,
    including proper backbone initialization and weight loading.

    Args:
        variant (str): The SegFormer variant to use (e.g., "SegFormerB0", "SegFormerB5").
            This determines the architecture configuration.

        backbone (keras.Model, optional): A pre-configured backbone model to use.
            If provided, must be initialized with `as_backbone=True`.
            If None, a MiT backbone corresponding to the variant will be created.
            Default: None

        num_classes (int, optional): Number of output classes for segmentation.
            Required unless using dataset-specific weights ("cityscapes" or "ade20k").
            Default: None (will be set based on weights if using dataset-specific weights)

        input_shape (tuple, optional): Input shape in format (height, width, channels).
            If None, the size will be determined based on the weights.
            Default: None

        input_tensor (Tensor, optional): Optional input tensor to use instead of creating
            a new input layer. Useful for connecting this model to other models.
            Default: None

        weights (str or None, optional): Pre-trained weights to use. Options:
            - "mit": Use ImageNet pre-trained MiT backbone weights only
            - "cityscapes_1024": Use weights pre-trained on Cityscapes dataset with 1024px resolution
            - "cityscapes_768": Use weights pre-trained on Cityscapes dataset with 768px resolution
            - "ade20k_512": Use weights pre-trained on ADE20K dataset with 512px resolution
            - "ade20k_640": Use weights pre-trained on ADE20K dataset with 640px resolution
            - None: No pre-trained weights
            - Path to a weights file: Load weights from specified file
            Default: "mit"

        **kwargs: Additional keyword arguments passed to the SegFormer constructor.

    Returns:
        keras.Model: Configured SegFormer model with requested architecture and weights.

    Raises:
        ValueError: If invalid weights are specified, if num_classes is not provided when
                    needed, or if an invalid backbone is provided.
    """

    DATASET_DEFAULT_CLASSES = {
        "ade20k": 150,
        "cityscapes": 19,
    }

    valid_model_weights = []
    if variant in SEGFORMER_WEIGHTS_CONFIG:
        valid_model_weights = list(SEGFORMER_WEIGHTS_CONFIG[variant].keys())

    valid_weights = [None, "mit"] + valid_model_weights

    if weights not in valid_weights and not isinstance(weights, str):
        raise ValueError(
            f"Invalid weights: {weights}. "
            f"Supported weights for {variant} are {', '.join([str(w) for w in valid_weights])}, "
            "a path to a weights file, or None."
        )

    weight_dataset = None
    original_num_classes = None
    if isinstance(weights, str):
        if "cityscapes" in weights:
            weight_dataset = "cityscapes"
            original_num_classes = DATASET_DEFAULT_CLASSES["cityscapes"]
        elif "ade20k" in weights:
            weight_dataset = "ade20k"
            original_num_classes = DATASET_DEFAULT_CLASSES["ade20k"]

    if num_classes is None and original_num_classes is not None:
        num_classes = original_num_classes
        print(
            f"No num_classes specified. Using {num_classes} classes from the {weight_dataset} dataset."
        )
    elif num_classes is None:
        raise ValueError(
            "num_classes must be specified when not using dataset-specific weights."
        )

    use_original_classes = (
        (weights in valid_model_weights)
        and (num_classes != original_num_classes)
        and (original_num_classes is not None)
    )
    model_num_classes = original_num_classes if use_original_classes else num_classes

    if input_shape is None:
        default_height, default_width = 512, 512

        if isinstance(weights, str):
            if "1024" in weights:
                default_height, default_width = 1024, 1024
            elif "768" in weights:
                default_height, default_width = 768, 768
            elif "640" in weights:
                default_height, default_width = 640, 640
            elif "512" in weights:
                default_height, default_width = 512, 512

        input_shape = (default_height, default_width, 3)
        print(
            f"Using default input shape {input_shape} based on weights configuration."
        )

    if (
        isinstance(weights, str)
        and weight_dataset
        and num_classes != DATASET_DEFAULT_CLASSES[weight_dataset]
    ):
        print(
            f"Using {weight_dataset} weights ({DATASET_DEFAULT_CLASSES[weight_dataset]} classes) "
            f"for fine-tuning to {num_classes} classes."
        )

    mit_variant = variant.replace("SegFormer", "MiT_")

    if backbone is None:
        backbone_function = getattr(mit, mit_variant)

        if weights == "mit":
            backbone_weights = "in1k"
            print(
                f"No backbone specified. "
                f"Using {mit_variant} backbone with ImageNet-1K (in1k) weights by default."
            )
        else:
            backbone_weights = None
            if weights is None:
                print(
                    f"No backbone specified and no weights provided. "
                    f"Using {mit_variant} backbone with no pre-trained weights."
                )
            else:
                print(
                    f"Using {mit_variant} backbone with no pre-trained weights since "
                    f"{weights} segmentation weights will be loaded."
                )

        backbone = backbone_function(
            include_top=False,
            as_backbone=True,
            input_shape=input_shape,
            weights=backbone_weights,
            include_normalization=False,
        )
    else:
        if not getattr(backbone, "as_backbone", False):
            raise ValueError(
                "The provided backbone must be initialized with as_backbone=True"
            )
        print(f"Using custom backbone provided by user for {variant}.")

    model = SegFormer(
        **SEGFORMER_MODEL_CONFIG[variant],
        backbone=backbone,
        num_classes=model_num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=variant,
        **kwargs,
    )

    if weights in valid_model_weights:
        print(f"Loading {weights} weights for {variant}.")
        load_weights_from_config(variant, weights, model, SEGFORMER_WEIGHTS_CONFIG)
    elif (
        weights is not None
        and weights != "mit"
        and isinstance(weights, str)
        and not weights.startswith(("cityscapes", "ade20k"))
    ):
        print(f"Loading weights from file: {weights}")
        model.load_weights(weights)
    elif weights == "mit":
        pass
    else:
        print("No weights loaded for the segmentation model.")

    if use_original_classes and num_classes != original_num_classes:
        print(
            f"Modifying classifier head from {original_num_classes} to {num_classes} classes."
        )

        new_model = SegFormer(
            **SEGFORMER_MODEL_CONFIG[variant],
            backbone=backbone,
            num_classes=num_classes,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=variant,
            **kwargs,
        )

        backbone_trained = model.backbone

        new_model.backbone.set_weights(backbone_trained.get_weights())

        head_prefix = f"{variant}_head"
        head_layers_old = [
            layer
            for layer in model.layers
            if layer.name.startswith(head_prefix)
            and not layer.name.endswith("classifier")
        ]
        head_layers_new = [
            layer
            for layer in new_model.layers
            if layer.name.startswith(head_prefix)
            and not layer.name.endswith("classifier")
        ]

        for layer_old, layer_new in zip(head_layers_old, head_layers_new):
            if layer_old.name == layer_new.name:
                layer_new.set_weights(layer_old.get_weights())

        return new_model

    return model


@register_model
def SegFormerB0(
    backbone=None,
    num_classes=None,
    input_shape=None,
    input_tensor=None,
    weights="mit",
    **kwargs,
):
    return _create_segformer_model(
        "SegFormerB0",
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def SegFormerB1(
    backbone=None,
    num_classes=None,
    input_shape=None,
    input_tensor=None,
    weights="mit",
    **kwargs,
):
    return _create_segformer_model(
        "SegFormerB1",
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def SegFormerB2(
    backbone=None,
    num_classes=None,
    input_shape=None,
    input_tensor=None,
    weights="mit",
    **kwargs,
):
    return _create_segformer_model(
        "SegFormerB2",
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def SegFormerB3(
    backbone=None,
    num_classes=None,
    input_shape=None,
    input_tensor=None,
    weights="mit",
    **kwargs,
):
    return _create_segformer_model(
        "SegFormerB3",
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def SegFormerB4(
    backbone=None,
    num_classes=None,
    input_shape=None,
    input_tensor=None,
    weights="mit",
    **kwargs,
):
    return _create_segformer_model(
        "SegFormerB4",
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def SegFormerB5(
    backbone=None,
    num_classes=None,
    input_shape=None,
    input_tensor=None,
    weights="mit",
    **kwargs,
):
    return _create_segformer_model(
        "SegFormerB5",
        backbone=backbone,
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )
