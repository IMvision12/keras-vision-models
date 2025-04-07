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
    input_shape=(512, 512, 3),
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
            Only used when creating a new backbone.
            Default: (512, 512, 3)
        
        input_tensor (Tensor, optional): Optional input tensor to use instead of creating
            a new input layer. Useful for connecting this model to other models.
            Default: None
        
        weights (str or None, optional): Pre-trained weights to use. Options:
            - "mit": Use ImageNet pre-trained MiT backbone weights only
            - "cityscapes": Use weights pre-trained on Cityscapes dataset (19 classes)
            - "ade20k": Use weights pre-trained on ADE20K dataset (150 classes)
            - None: No pre-trained weights
            - Path to a weights file: Load weights from specified file
            Default: "mit"
        
        **kwargs: Additional keyword arguments passed to the SegFormer constructor.
    
    Returns:
        keras.Model: Configured SegFormer model with requested architecture and weights.
    
    Raises:
        ValueError: If invalid weights are specified, if num_classes is not provided when
                    needed, or if an invalid backbone is provided.
    
    Examples:
        # Create SegFormerB0 with ImageNet pre-trained backbone for 10 classes
        model = _create_segformer_model("SegFormerB0", num_classes=10)
        
        # Create SegFormerB3 pre-trained on Cityscapes
        model = _create_segformer_model("SegFormerB3", weights="cityscapes")
        
        # Create SegFormerB5 with custom input shape and no pre-trained weights
        model = _create_segformer_model("SegFormerB5", num_classes=5, 
                                        input_shape=(1024, 1024, 3), weights=None)
    """
    
    DATASET_DEFAULT_CLASSES = {
        "ade20k": 150,
        "cityscapes": 19,
    }
    
    valid_weights = [None, "cityscapes", "ade20k", "mit"]
    if weights not in valid_weights and not isinstance(weights, str):
        raise ValueError(
            f"Invalid weights: {weights}. "
            f"Supported weights are {', '.join([str(w) for w in valid_weights])}, "
            "a path to a weights file, or None."
        )
    
    if num_classes is None:
        if weights in DATASET_DEFAULT_CLASSES:
            num_classes = DATASET_DEFAULT_CLASSES[weights]
            print(f"Setting num_classes to {num_classes} based on {weights} dataset.")
        else:
            raise ValueError(
                "num_classes must be specified when not using dataset-specific weights."
            )
    
    if weights in DATASET_DEFAULT_CLASSES and num_classes != DATASET_DEFAULT_CLASSES[weights]:
        print(
            f"Warning: Using {weights} weights with {num_classes} classes instead of "
            f"the default {DATASET_DEFAULT_CLASSES[weights]} classes. "
            f"This may require fine-tuning to achieve good results."
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
        num_classes=num_classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=variant,
        **kwargs,
    )
    
    if weights in get_all_weight_names(SEGFORMER_WEIGHTS_CONFIG):
        print(f"Loading {weights} weights for {variant}.")
        load_weights_from_config(
            variant, weights, model, SEGFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None and weights != "mit":
        print(f"Loading weights from file: {weights}")
        model.load_weights(weights)
    else:
        print("No segmentation model weights loaded.")
    
    return model

@register_model
def SegFormerB0(
    backbone=None,
    num_classes=None,
    input_shape=(512, 512, 3),
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
    input_shape=(512, 512, 3),
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
    input_shape=(512, 512, 3),
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
    input_shape=(512, 512, 3),
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
    input_shape=(512, 512, 3),
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
    input_shape=(512, 512, 3),
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