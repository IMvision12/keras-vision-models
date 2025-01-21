import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

def create_segformer_head(features, embed_dim=256, num_classes=19, dropout_rate=0.1, name="segformer_head"):
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
        x = layers.Dense(
            embed_dim,
            name=f"{name}_linear_c{i+1}"
        )(feature)

        x = layers.Resizing(
            height=target_height,
            width=target_width,
            interpolation="bilinear",
            name=f"{name}_resize_c{i+1}"
        )(x)
        projected_features.append(x)

    x = layers.Concatenate(axis=-1, name=f"{name}_concat")(projected_features[::-1])

    x = layers.Conv2D(
        filters=embed_dim,
        kernel_size=1,
        use_bias=False,
        name=f"{name}_fusion_conv"
    )(x)
    x = layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.9,
        name=f"{name}_fusion_bn"
    )(x)
    x = layers.Activation("relu", name=f"{name}_fusion_relu")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_dropout")(x)

    x = layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        name=f"{name}_classifier"
    )(x)

    return x

def create_segformer(
    backbone,
    num_classes,
    embed_dim=256,
    dropout_rate=0.1,
    input_shape=None,
    name="segformer"
):
    if input_shape is None:
        input_shape = backbone.input_shape[1:]

    inputs = layers.Input(shape=input_shape, name=f"{name}_input")

    features = backbone(inputs)

    x = create_segformer_head(
        features=features,
        embed_dim=embed_dim,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        name=f"{name}_head"
    )

    x = layers.Resizing(
        height=input_shape[0],
        width=input_shape[1],
        interpolation="bilinear",
        name=f"{name}_final_upsampling"
    )(x)

    outputs = layers.Activation(
        "softmax",
        name=f"{name}_output_activation"
    )(x)

    model = layers.Model(
        inputs=inputs,
        outputs=outputs,
        name=name
    )

    return model


model = create_segformer(
        backbone=backbone,
        num_classes=150,
        embed_dim=256,
        dropout_rate=0.1,
        input_shape=(512,512,3),
        name="segformer_ade20k"
    )