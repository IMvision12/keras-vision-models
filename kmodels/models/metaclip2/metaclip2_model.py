import keras
from keras import layers, ops

from kmodels.model_registry import register_model
from kmodels.models.clip.clip_layers import (
    CLIPAttention,
    CLIPLogitScale,
    TextModelEmbedding,
    VisionModelEmbedding,
)
from kmodels.weight_utils import get_all_weight_names, load_weights_from_config

from .config import (
    METACLIP2_HF_CONVERT_DEFAULT_ALIAS,
    METACLIP2_HF_CONVERT_VARIANTS,
    METACLIP2_MODEL_CONFIG,
    METACLIP2_WEIGHTS_CONFIG,
)
from .metaclip2_tokenizer import METACLIP2_EOS_TOKEN_ID


def quick_gelu(x):
    return x * ops.sigmoid(1.702 * x)


def _activation_layer(hidden_act):
    if hidden_act == "quick_gelu":
        return keras.layers.Lambda(quick_gelu)
    return keras.layers.Activation(hidden_act)


def residual_attention_block(
    x,
    proj_dim,
    num_heads,
    layer_name_prefix,
    layer_idx,
    causal_attention_mask=None,
    attention_mask=None,
    mlp_ratio=4.0,
    hidden_act="gelu",
):
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
    mlp_output = _activation_layer(hidden_act)(mlp_output)
    mlp_output = keras.layers.Dense(proj_dim, name=f"{layer_prefix}_dense_2")(
        mlp_output
    )

    output = keras.layers.Add()([residual_1, mlp_output])
    return output


def metaclip2_encoder(
    inputs,
    width,
    num_layers,
    heads,
    layer_prefix=None,
    causal_attention_mask=None,
    attention_mask=None,
    mlp_ratio=None,
    hidden_act="gelu",
):
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
            hidden_act=hidden_act,
        )
    return x


def metaclip2_image_encoder(
    inputs,
    input_resolution=224,
    patch_size=16,
    width=768,
    num_layers=12,
    heads=12,
    output_dim=512,
    vision_mlp_ratio=4.0,
    hidden_act="gelu",
    data_format="channels_last",
):
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
    encoded = metaclip2_encoder(
        x,
        width=width,
        num_layers=num_layers,
        heads=heads,
        layer_prefix="vision_model_encoder",
        mlp_ratio=vision_mlp_ratio,
        hidden_act=hidden_act,
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


def metaclip2_text_encoder(
    inputs,
    attention_mask,
    transformer_width,
    transformer_layers,
    transformer_heads,
    vocab_size,
    embed_dim,
    context_length,
    text_mlp_ratio,
    hidden_act="gelu",
    eos_token_id=METACLIP2_EOS_TOKEN_ID,
):
    x = TextModelEmbedding(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=transformer_width,
        name="text_model_embedding",
    )(inputs)

    causal_attention_mask = ops.triu(
        ops.ones((context_length, context_length)) * (-1e8), k=1
    )

    attention_mask_float = ops.cast(attention_mask, dtype="float32")
    expanded_mask = ops.reshape(attention_mask_float, (-1, 1, 1, context_length))
    expanded_mask = ops.repeat(expanded_mask, context_length, axis=2)
    expanded_mask = (1.0 - expanded_mask) * (-1e8)

    encoded_output = metaclip2_encoder(
        x,
        width=transformer_width,
        num_layers=transformer_layers,
        heads=transformer_heads,
        causal_attention_mask=causal_attention_mask,
        attention_mask=expanded_mask,
        mlp_ratio=text_mlp_ratio,
        hidden_act=hidden_act,
        layer_prefix="text_model_encoder",
    )

    layer_norm = keras.layers.LayerNormalization(name="text_model_layernorm")(
        encoded_output
    )

    eos_mask = ops.cast(ops.equal(inputs, eos_token_id), "int32")
    indices = ops.argmax(eos_mask, axis=-1)

    one_hot_indices = ops.one_hot(indices, context_length)
    selected_features = ops.einsum("bi,bij->bj", one_hot_indices, layer_norm)
    selected_features = ops.expand_dims(selected_features, axis=1)

    text_features = keras.layers.Dense(
        embed_dim, name="text_projection", use_bias=False
    )(selected_features)

    output = ops.squeeze(text_features, axis=1)
    return output


def metaclip2_head(image_embeddings, text_embeddings):
    normalize_image_features = ops.sqrt(
        ops.sum(ops.power(image_embeddings, 2), axis=-1, keepdims=True)
    )
    normalize_text_features = ops.sqrt(
        ops.sum(ops.power(text_embeddings, 2), axis=-1, keepdims=True)
    )
    image_embeddings = image_embeddings / normalize_image_features
    text_embeddings = text_embeddings / normalize_text_features
    logit_scale_layer = CLIPLogitScale(initial_value=0.07, name="logit_scale")
    image_logits, text_logits = logit_scale_layer([image_embeddings, text_embeddings])
    return image_logits, text_logits


@keras.saving.register_keras_serializable(package="kmodels")
class MetaClip2Model(keras.Model):
    """MetaCLIP 2 (multilingual / worldwide) contrastive vision-language model.

    MetaCLIP 2 is Meta's 2nd-generation CLIP, trained on multilingual data with
    the XLM-R tokenizer (vocab 901629). Architecturally it is identical to
    OpenAI CLIP except for:

    - Configurable MLP activation (``"gelu"`` or ``"quick_gelu"``).
    - EOS pooling uses explicit ``eos_token_id == 2`` match instead of
      argmax-over-token-ids (needed because mask_token_id > eos_token_id).
    - Wider / deeper text tower in larger variants.

    Reference:
      - https://arxiv.org/abs/2507.22062 ("MetaCLIP 2")
      - https://huggingface.co/docs/transformers/model_doc/metaclip_2
    """

    def __init__(
        self,
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        vision_heads=None,
        context_length=77,
        vocab_size=901629,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        vision_mlp_ratio=4.0,
        text_mlp_ratio=4.0,
        hidden_act="gelu",
        eos_token_id=METACLIP2_EOS_TOKEN_ID,
        input_shape=None,
        input_tensor=None,
        weights="worldwide_224",
        name="MetaClip2Model",
        **kwargs,
    ):
        if vision_heads is None:
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
                channels = input_shape[2] if len(input_shape) == 3 else 3
        else:
            image_size = image_resolution
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

        image_embeddings = metaclip2_image_encoder(
            images_input,
            input_resolution=image_size,
            patch_size=vision_patch_size,
            width=vision_width,
            num_layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            vision_mlp_ratio=vision_mlp_ratio,
            hidden_act=hidden_act,
            data_format=data_format,
        )

        text_embeddings = metaclip2_text_encoder(
            token_ids_input,
            attention_mask=padding_mask_input,
            transformer_width=transformer_width,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            text_mlp_ratio=text_mlp_ratio,
            context_length=context_length,
            hidden_act=hidden_act,
            eos_token_id=eos_token_id,
        )

        image_logits, text_logits = metaclip2_head(image_embeddings, text_embeddings)

        outputs = {"image_logits": image_logits, "text_logits": text_logits}
        inputs = {
            "images": images_input,
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.embed_dim = embed_dim
        self.image_resolution = image_resolution
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.vision_heads = vision_heads
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.vision_mlp_ratio = vision_mlp_ratio
        self.text_mlp_ratio = text_mlp_ratio
        self.hidden_act = hidden_act
        self.eos_token_id = eos_token_id
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
                "image_resolution": self.image_resolution,
                "input_shape": image_input_shape,
                "vision_layers": self.vision_layers,
                "vision_width": self.vision_width,
                "vision_patch_size": self.vision_patch_size,
                "vision_heads": self.vision_heads,
                "context_length": self.context_length,
                "vocab_size": self.vocab_size,
                "transformer_width": self.transformer_width,
                "transformer_heads": self.transformer_heads,
                "transformer_layers": self.transformer_layers,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "text_mlp_ratio": self.text_mlp_ratio,
                "hidden_act": self.hidden_act,
                "eos_token_id": self.eos_token_id,
                "input_tensor": self.input_tensor,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_metaclip2(
    variant,
    weights,
    input_tensor,
    input_shape,
    name,
    **kwargs,
):
    cfg = {**METACLIP2_MODEL_CONFIG[variant], **kwargs}
    model = MetaClip2Model(
        **cfg,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        name=name,
    )

    if (
        variant in METACLIP2_HF_CONVERT_VARIANTS
        and weights == METACLIP2_HF_CONVERT_DEFAULT_ALIAS.get(variant)
    ):
        from kmodels.models.metaclip2.convert_metaclip2_hf_to_keras import (
            transfer_metaclip2_weights,
        )
        from kmodels.weight_utils.hf_gated_weight_download import (
            load_and_convert_from_hf,
        )

        load_and_convert_from_hf(
            model=model,
            model_name=variant.lower(),
            hf_model_id=METACLIP2_HF_CONVERT_VARIANTS[variant],
            transfer_fn=transfer_metaclip2_weights,
        )
    elif weights in get_all_weight_names(METACLIP2_WEIGHTS_CONFIG):
        load_weights_from_config(variant, weights, model, METACLIP2_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)

    return model


@register_model
def MetaClip2WorldwideS16(
    weights="worldwide_224",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2WorldwideS16",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2WorldwideS16",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2WorldwideS16_384(
    weights="worldwide_384",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2WorldwideS16_384",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2WorldwideS16_384",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2WorldwideM16(
    weights="worldwide_224",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2WorldwideM16",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2WorldwideM16",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2WorldwideM16_384(
    weights="worldwide_384",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2WorldwideM16_384",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2WorldwideM16_384",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2WorldwideB16(
    weights="worldwide_224",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2WorldwideB16",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2WorldwideB16",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2WorldwideB16_384(
    weights="worldwide_384",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2WorldwideB16_384",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2WorldwideB16_384",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2WorldwideB32(
    weights="worldwide_224",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2WorldwideB32",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2WorldwideB32",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2WorldwideB32_384(
    weights="worldwide_384",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2WorldwideB32_384",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2WorldwideB32_384",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2WorldwideL14(
    weights="worldwide_224",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2WorldwideL14",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2WorldwideL14",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2WorldwideHugeQuickgelu(
    weights="worldwide_224",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2WorldwideHugeQuickgelu",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2WorldwideHugeQuickgelu",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2WorldwideHuge378(
    weights="worldwide_378",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2WorldwideHuge378",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2WorldwideHuge378",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2WorldwideGiant(
    weights="worldwide_224",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2WorldwideGiant",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2WorldwideGiant",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2WorldwideGiant378(
    weights="worldwide_378",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2WorldwideGiant378",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2WorldwideGiant378",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2Mt5WorldwideS16(
    weights="mt5_worldwide_224",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2Mt5WorldwideS16",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2Mt5WorldwideS16",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2Mt5WorldwideM16(
    weights="mt5_worldwide_224",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2Mt5WorldwideM16",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2Mt5WorldwideM16",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )


@register_model
def MetaClip2Mt5WorldwideB32(
    weights="mt5_worldwide_224",
    input_tensor=None,
    input_shape=None,
    name="MetaClip2Mt5WorldwideB32",
    **kwargs,
):
    return _create_metaclip2(
        "MetaClip2Mt5WorldwideB32",
        weights,
        input_tensor,
        input_shape,
        name,
        **kwargs,
    )
