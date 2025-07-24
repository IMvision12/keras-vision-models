import keras
from keras import layers, ops

from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .clip_layers import (
    CLIPAttention,
    CLIPLogitScale,
    TextModelEmbedding,
    VisionModelEmbedding,
)
from .config import CLIP_MODEL_CONFIG, CLIP_WEIGHTS_CONFIG


def quick_gelu(x):
    """Applies the Quick GELU activation function to the input tensor.

    This is an approximation of the GELU (Gaussian Error Linear Unit)
    activation function that uses the sigmoid function for more efficient
    computation. It's used in the CLIP model as a replacement for the
    standard GELU activation.

    Args:
        x: Input tensor.

    Returns:
        Tensor with Quick GELU activation applied.
    """
    return x * ops.sigmoid(1.702 * x)


def residual_attention_block(
    x,
    proj_dim,
    num_heads,
    layer_name_prefix,
    layer_idx,
    causal_attention_mask=None,
    attention_mask=None,
    mlp_ratio=4.0,
):
    """Creates a residual attention block used in the CLIP transformer encoder.

    This function implements a standard transformer block consisting of multi-head
    self-attention followed by a feed-forward MLP block, with layer normalization
    and residual connections. The same architecture is used in both the vision and
    text encoders of the CLIP model.

    Args:
        x: Input tensor of shape (batch_size, sequence_length, proj_dim).
        proj_dim: Integer, dimensionality of the projection space (hidden size).
        num_heads: Integer, number of attention heads.
        layer_name_prefix: String, prefix for naming layers (e.g., "text_model_encoder"
            or "vision_model_encoder").
        layer_idx: Integer, index of the current layer in the encoder.
        causal_attention_mask: Optional tensor of shape (sequence_length, sequence_length)
            used for causal (autoregressive) attention in the text encoder.
        attention_mask: Optional tensor used to mask padding tokens in text processing.
        mlp_ratio: Optional float, ratio of the MLP hidden dimension to the embedding
            dimension. The MLP expands the representation by this factor in its hidden
            layer. If None, defaults to 4.0.

    Returns:
        Output tensor of shape (batch_size, sequence_length, proj_dim).

    """
    layer_prefix = f"{layer_name_prefix}_{layer_idx}"

    ln_1_output = keras.layers.LayerNormalization(
        epsilon=1e-5, name=f"{layer_prefix}_layernorm_1"
    )(x)

    mask = None
    if causal_attention_mask is not None:
        mask = ops.cast(causal_attention_mask, dtype=x.dtype)
    if attention_mask is not None:
        attention_mask = ops.cast(attention_mask, dtype=x.dtype)
        mask = (
            ops.add(causal_attention_mask, attention_mask)
            if causal_attention_mask is not None
            else attention_mask
        )

    attention_output = CLIPAttention(
        proj_dim=proj_dim,
        num_heads=num_heads,
        name_prefix=f"{layer_prefix}_attn",
    )(ln_1_output, attention_mask=mask)[0]

    residual_1 = keras.layers.Add()([x, attention_output])
    ln_2_output = keras.layers.LayerNormalization(
        epsilon=1e-5, name=f"{layer_prefix}_layernorm_2"
    )(residual_1)

    mlp_intermediate_size = int(proj_dim * mlp_ratio)
    mlp_output = keras.layers.Dense(
        mlp_intermediate_size, name=f"{layer_prefix}_dense_1"
    )(ln_2_output)
    mlp_output = keras.layers.Lambda(quick_gelu)(mlp_output)
    mlp_output = keras.layers.Dense(proj_dim, name=f"{layer_prefix}_dense_2")(
        mlp_output
    )

    output = keras.layers.Add()([residual_1, mlp_output])

    return output


def clip_encoder(
    inputs,
    width,
    num_layers,
    heads,
    layer_prefix=None,
    causal_attention_mask=None,
    attention_mask=None,
    mlp_ratio=None,
):
    """Creates a transformer encoder used in both vision and text components of CLIP.

    This function implements a standard transformer encoder architecture that is shared
    between the vision and text branches of CLIP, with minor differences handled through
    parameters. The encoder consists of a sequence of residual attention blocks, each
    containing multi-head self-attention followed by an MLP block with layer normalization
    and residual connections.

    Args:
        inputs: Tensor of shape (batch_size, sequence_length, width) containing the
            embedded input sequence (either text tokens or image patches).
        width: Integer, dimensionality of the transformer's hidden representations.
        num_layers: Integer, number of transformer layers in the encoder.
        heads: Integer, number of attention heads in each transformer layer. Should
            typically be width / 64 for optimal performance.
        layer_prefix: Optional string, prefix for naming layers to distinguish between
            vision and text encoders when sharing the same architecture.
        causal_attention_mask: Optional tensor of shape (sequence_length, sequence_length)
            used for causal (autoregressive) attention in the text encoder. Set to None
            for the vision encoder.
        attention_mask: Optional tensor used to mask padding tokens in text processing.
            Set to None for the vision encoder.
        mlp_ratio: Optional float, ratio of the MLP hidden dimension to the embedding
            dimension. The MLP expands the representation by this factor in its hidden
            layer. If None, defaults to 4.0.

    Returns:
        A tensor of shape (batch_size, sequence_length, width) containing the encoded
        representations of the input sequence.

    """
    x = inputs

    for i in range(num_layers):
        x = residual_attention_block(
            x,
            proj_dim=width,
            num_heads=heads,
            layer_name_prefix=layer_prefix,
            layer_idx=i,
            causal_attention_mask=causal_attention_mask,
            attention_mask=attention_mask,
            mlp_ratio=mlp_ratio,
        )

    return x


def clip_image_encoder(
    inputs,
    input_resolution=224,
    patch_size=16,
    width=768,
    num_layers=12,
    heads=12,
    output_dim=512,
    vision_mlp_ratio=4.0,
    data_format="channels_last",
):
    """Creates a CLIP image encoder based on Vision Transformer (ViT) architecture.

    This function implements the vision component of the CLIP model, which processes
    images using a Vision Transformer (ViT) architecture. The encoder divides the input
    image into fixed-size patches, linearly embeds each patch, adds position embeddings,
    and processes the resulting sequence with a transformer encoder. The final
    representation is obtained from the class token and projected to the joint
    embedding space.

    Args:
        inputs: Tensor of shape (batch_size, height, width, channels) containing the
            input images.
        input_resolution: Integer, resolution of input images (both height and width).
            Images are expected to be square with this resolution.
        patch_size: Integer, size of image patches. The image will be divided into
            patches of this size, which determines the sequence length for the transformer.
        width: Integer, dimensionality of the transformer's hidden representations.
        num_layers: Integer, number of transformer layers in the vision encoder.
        heads: Integer, number of attention heads in each transformer layer.
        output_dim: Integer, dimensionality of the final image embedding output that
            matches the joint embedding space.
        vision_mlp_ratio: Float, ratio of the MLP hidden dimension to the embedding
            dimension in the vision transformer. The MLP expands the representation
            by this factor in its hidden layer. Default is 4.0.
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.

    Returns:
        A tensor of shape (batch_size, output_dim) containing the image embeddings
        that can be compared with text embeddings in the joint embedding space.

    """
    patch_embeddings = keras.layers.Conv2D(
        filters=width,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="vision_model_conv",
    )(inputs)

    embeddings = VisionModelEmbedding(
        width, input_resolution, patch_size, data_format, name="vision_model_embeddings"
    )(patch_embeddings)

    x = keras.layers.LayerNormalization(epsilon=1e-5, name="vision_model_layernorm_1")(
        embeddings
    )
    encoded = clip_encoder(
        x,
        width=width,
        num_layers=num_layers,
        heads=heads,
        layer_prefix="vision_model_encoder",
        mlp_ratio=vision_mlp_ratio,
    )

    class_token = keras.layers.Lambda(lambda x: x[:, 0, :], name="extract_token")(
        encoded
    )
    x = keras.layers.LayerNormalization(epsilon=1e-5, name="vision_model_layernorm_2")(
        class_token
    )
    outputs = keras.layers.Dense(output_dim, use_bias=False, name="visual_projection")(
        x
    )

    return outputs


def clip_text_encoder(
    inputs,
    attention_mask,
    transformer_width,
    transformer_layers,
    transformer_heads,
    vocab_size,
    embed_dim,
    context_length,
    text_mlp_ratio,
):
    """Creates a CLIP text encoder for processing tokenized text inputs.

    This function implements the text encoder component of the CLIP architecture,
    which consists of a transformer model with token embeddings, positional
    embeddings, masked self-attention, and a final projection to the joint
    embedding space. The text encoder processes tokenized text inputs and
    produces embeddings that can be compared with image embeddings.

    Args:
        inputs: Tensor of shape (batch_size, context_length) containing tokenized
            text sequences with padding as needed.
        attention_mask: Tensor of shape (batch_size, context_length) containing
            1s for non-padding tokens and 0s for padding tokens.
        transformer_width: Integer, dimensionality of the transformer's hidden
            representations.
        transformer_layers: Integer, number of transformer layers in the text encoder.
        transformer_heads: Integer, number of attention heads in each transformer layer.
        vocab_size: Integer, size of the token vocabulary.
        embed_dim: Integer, dimensionality of the final text embedding output.
        context_length: Integer, maximum length of input text sequences.
        text_mlp_ratio: Float, ratio of the MLP hidden dimension to the embedding
            dimension in the text transformer. The MLP expands the representation
            by this factor in its hidden layer. Default is 4.0.

    Returns:
        A tensor of shape (batch_size, embed_dim) containing the text embeddings
        that can be compared with image embeddings in the joint embedding space.

    """
    x = TextModelEmbedding(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=transformer_width,
        name="text_model_embedding",
    )(inputs)

    causal_attention_mask = ops.cast(
        ops.triu(ops.ones((context_length, context_length))), "float32"
    )

    attention_mask_float = ops.cast(attention_mask, dtype="float32")
    expanded_mask = ops.reshape(attention_mask_float, (-1, 1, 1, context_length))
    expanded_mask = ops.repeat(expanded_mask, context_length, axis=2)
    expanded_mask = (1.0 - expanded_mask) * (-1e8)

    encoded_output = clip_encoder(
        x,
        width=transformer_width,
        num_layers=transformer_layers,
        heads=transformer_heads,
        causal_attention_mask=causal_attention_mask,
        attention_mask=expanded_mask,
        mlp_ratio=text_mlp_ratio,
        layer_prefix="text_model_encoder",
    )

    layer_norm = keras.layers.LayerNormalization(name="text_model_layernorm")(
        encoded_output
    )

    indices = ops.argmax(inputs, axis=-1)

    one_hot_indices = ops.one_hot(indices, context_length)
    selected_features = ops.einsum("bi,bij->bj", one_hot_indices, layer_norm)
    selected_features = ops.expand_dims(selected_features, axis=1)

    text_features = keras.layers.Dense(
        embed_dim, name="text_projection", use_bias=False
    )(selected_features)

    output = ops.squeeze(text_features, axis=1)
    return output


def clip_head(image_embeddings, text_embeddings):
    """Creates the CLIP model head that processes embedded image and text features.

    This function performs normalization of the image and text embeddings and applies
    a learned temperature parameter (logit scale) to control the sharpness of the
    similarity distribution. The normalization ensures that similarity is measured
    by cosine distance, while the temperature scaling helps with training stability
    and convergence.

    Args:
        image_embeddings: Tensor of shape (batch_size, embed_dim) containing the
            output features from the image encoder.
        text_embeddings: Tensor of shape (batch_size, embed_dim) containing the
            output features from the text encoder.

    Returns:
        A tuple of (image_logits, text_logits):
            - image_logits: Normalized and scaled image embeddings of shape
              (batch_size, embed_dim)
            - text_logits: Normalized and scaled text embeddings of shape
              (batch_size, embed_dim)

    """
    normalize_image_features = ops.sqrt(
        ops.sum(ops.power(image_embeddings, 2), keepdims=True)
    )
    normalize_text_features = ops.sqrt(
        ops.sum(ops.power(text_embeddings, 2), keepdims=True)
    )
    image_embeddings = image_embeddings / normalize_image_features
    text_embeddings = text_embeddings / normalize_text_features
    logit_scale_layer = CLIPLogitScale(initial_value=0.07, name="logit_scale")
    image_logits, text_logits = logit_scale_layer([image_embeddings, text_embeddings])
    return image_logits, text_logits


@keras.saving.register_keras_serializable(package="kvmm")
class CLIPModel(keras.Model):
    """Instantiates the Contrastive Language-Image Pre-training (CLIP) architecture.

    CLIP is a neural network trained on a variety of (image, text) pairs. It can be used
    for zero-shot image classification, image-text similarity ranking, and other
    multimodal tasks. This implementation follows the original paper architecture.

    Reference:
    - [Learning Transferable Visual Models From Natural Language Supervision](
        https://arxiv.org/abs/2103.00020) (Radford et al., ICML 2021)
    - [Improving Vision-Language Pre-training with Large-scale Caption Annotations](
        https://arxiv.org/abs/2111.08735)
    - [OpenAI CLIP GitHub Repository](https://github.com/openai/CLIP)

    The model consists of two main components:
    1. A vision transformer (ViT) that encodes images
    2. A text transformer that encodes text

    These encoders project images and text into a shared embedding space where
    similarity is computed using cosine similarity.

    Args:
        embed_dim: Integer, dimensionality of the final joint embedding space where
            image and text features are projected.
        image_resolution: Integer, resolution of input images (both height and width).
            Images will be resized to this resolution before processing.
        vision_layers: Integer, number of transformer layers in the vision model.
            Deeper models generally perform better but require more computation.
        vision_width: Integer, width/dimensionality of the vision transformer's hidden
            representations.
        vision_patch_size: Integer, size of patches for the vision transformer. The image
            will be divided into patches of this size before processing.
        context_length: Integer, maximum length of input text sequences. Longer sequences
            will be truncated.
        vocab_size: Integer, size of the token vocabulary for text processing.
        transformer_width: Integer, width/dimensionality of the text transformer's hidden
            representations.
        transformer_heads: Integer, number of attention heads in the text transformer.
            Should typically be transformer_width / 64.
        transformer_layers: Integer, number of transformer layers in the text transformer.
        vision_mlp_ratio: Float, ratio of the MLP hidden dimension to the embedding
            dimension in the vision transformer. The MLP expands the representation
            by this factor in its hidden layer. Default is 4.0.
        text_mlp_ratio: Float, ratio of the MLP hidden dimension to the embedding
            dimension in the text transformer. The MLP expands the representation
            by this factor in its hidden layer. Default is 4.0.
        input_tensor: Optional Keras tensor (output of `layers.Input()`) to use as
            the model's input. If not provided, new input tensors are created.
        name: String, the name of the model. Defaults to `"CLIPModel"`.
        **kwargs: Additional keyword arguments passed to the base class.

    Returns:
        A Keras `Model` instance with image and text inputs, and embedding outputs.

    Note:
        - Both image and text features are L2-normalized before output
        - The model can be used for zero-shot classification by comparing image
          embeddings with text embeddings of class descriptions
        - For best performance with pretrained weights, use the same preprocessing
          as was used during training
    """

    def __init__(
        self,
        embed_dim=512,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        vision_mlp_ratio=4.0,
        text_mlp_ratio=4.0,
        input_shape=None,
        input_tensor=None,
        weights="openai_224",
        name="CLIPModel",
        **kwargs,
    ):
        vision_heads = vision_width // 64
        data_format = keras.backend.image_data_format()

        if input_shape is not None:
            if data_format == "channels_first":
                if len(input_shape) == 3:
                    channels = input_shape[0]
                    image_size = min(input_shape[1], input_shape[2])
                else:
                    channels = 3
                    image_size = input_shape[0] if len(input_shape) >= 1 else 224
            else:
                if len(input_shape) >= 2:
                    image_size = min(input_shape[0], input_shape[1])
                else:
                    image_size = input_shape[0] if len(input_shape) >= 1 else 224

                if len(input_shape) == 3:
                    channels = input_shape[2]
                else:
                    channels = 3
        else:
            image_size = 336 if weights and "336" in weights else 224
            channels = 3

        if data_format == "channels_first":
            image_input_shape = [channels, image_size, image_size]
        else:
            image_input_shape = [image_size, image_size, channels]

        if isinstance(input_tensor, dict):
            images_input = input_tensor.get("images") or layers.Input(
                shape=image_input_shape, name="images"
            )
            token_ids_input = input_tensor.get("token_ids") or layers.Input(
                shape=[context_length], name="token_ids"
            )
            padding_mask_input = input_tensor.get("padding_mask") or layers.Input(
                shape=[context_length], name="padding_mask"
            )
        else:
            images_input = layers.Input(shape=image_input_shape, name="images")
            token_ids_input = layers.Input(shape=[context_length], name="token_ids")
            padding_mask_input = layers.Input(
                shape=[context_length], name="padding_mask"
            )

        image_embeddings = clip_image_encoder(
            images_input,
            input_resolution=image_size,
            patch_size=vision_patch_size,
            width=vision_width,
            num_layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            vision_mlp_ratio=vision_mlp_ratio,
            data_format=data_format,
        )

        text_embeddings = clip_text_encoder(
            token_ids_input,
            attention_mask=padding_mask_input,
            transformer_width=transformer_width,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            text_mlp_ratio=text_mlp_ratio,
            context_length=context_length,
        )

        image_logits, text_logits = clip_head(
            image_embeddings,
            text_embeddings,
        )

        outputs = {
            "image_logits": image_logits,
            "text_logits": text_logits,
        }

        inputs = {
            "images": images_input,
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        # Store model parameters
        self.embed_dim = embed_dim
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.vision_mlp_ratio = vision_mlp_ratio
        self.text_mlp_ratio = text_mlp_ratio
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()

        image_shape_with_batch = self.input_shape[0]
        if image_shape_with_batch[0] is None:
            image_input_shape = image_shape_with_batch[1:]
        else:
            image_input_shape = image_shape_with_batch

        config.update(
            {
                "embed_dim": self.embed_dim,
                "input_shape": image_input_shape,
                "vision_layers": self.vision_layers,
                "vision_width": self.vision_width,
                "vision_patch_size": self.vision_patch_size,
                "context_length": self.context_length,
                "vocab_size": self.vocab_size,
                "transformer_width": self.transformer_width,
                "transformer_heads": self.transformer_heads,
                "transformer_layers": self.transformer_layers,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "text_mlp_ratio": self.text_mlp_ratio,
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
def ClipVitBase16(
    weights="openai_224",
    input_tensor=None,
    input_shape=None,
    name="ClipVitBase16",
    **kwargs,
):
    model = CLIPModel(
        **CLIP_MODEL_CONFIG["ClipVitBase16"],
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        weights=weights,
        **kwargs,
    )

    if weights in get_all_weight_names(CLIP_WEIGHTS_CONFIG):
        load_weights_from_config("ClipVitBase16", weights, model, CLIP_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ClipVitBase32(
    weights="openai_224",
    input_tensor=None,
    input_shape=None,
    name="ClipVitBase32",
    **kwargs,
):
    model = CLIPModel(
        **CLIP_MODEL_CONFIG["ClipVitBase32"],
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(CLIP_WEIGHTS_CONFIG):
        load_weights_from_config("ClipVitBase32", weights, model, CLIP_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ClipVitLarge14(
    weights="openai_224",
    input_tensor=None,
    input_shape=None,
    name="ClipVitLarge14",
    **kwargs,
):
    model = CLIPModel(
        **CLIP_MODEL_CONFIG["ClipVitLarge14"],
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(CLIP_WEIGHTS_CONFIG):
        load_weights_from_config("ClipVitLarge14", weights, model, CLIP_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ClipVitG14(
    weights="laion2b_s12B_b42K_224",
    input_tensor=None,
    input_shape=None,
    name="ClipVitG14",
    **kwargs,
):
    model = CLIPModel(
        **CLIP_MODEL_CONFIG["ClipVitG14"],
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(CLIP_WEIGHTS_CONFIG):
        load_weights_from_config("ClipVitG14", weights, model, CLIP_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ClipVitBigG14(
    weights="laion2b_39B_b160k_224",
    input_tensor=None,
    input_shape=None,
    name="ClipVitBigG14",
    **kwargs,
):
    model = CLIPModel(
        **CLIP_MODEL_CONFIG["ClipVitBigG14"],
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(CLIP_WEIGHTS_CONFIG):
        load_weights_from_config("ClipVitBigG14", weights, model, CLIP_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
