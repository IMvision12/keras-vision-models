"""Qwen2-VL model builders: LLM (text) + ViT vision tower + full multimodal.

Design:
  * ``build_qwen2_llm(cfg)`` — functional model accepting
    ``{"input_embeds", "position_ids", "causal_mask"}``; returns LLM hidden
    states. Accepting embeds (not ids) lets the multimodal path substitute
    vision features at the embedding layer.
  * ``build_qwen2_vision(cfg)`` — functional model accepting packed
    ``{"pixel_values", "grid_thw", "rotary_cos", "rotary_sin"}``; returns
    LLM-space hidden states (post-merger).
  * Per-variant factories (``Qwen2VL2B``, ``Qwen2VL2BInstruct``,
    ``Qwen2VL7B``, ``Qwen2VL7BInstruct``, ``Qwen2VL72B``,
    ``Qwen2VL72BInstruct``) — return a bundle dict
    ``{config, embed_tokens, llm, vision, lm_head?, llm_inv_freq,
    vision_inv_freq}``. ``lm_head`` is included only when the variant has
    ``tie_word_embeddings=False`` (7B / 72B). The 2B variants reuse
    ``embed_tokens`` for the output projection.
"""

import keras
import numpy as np
from keras import layers, ops

from kmodels.model_registry import register_model

from .config import QWEN2_VL_HF_CONVERT_VARIANTS, QWEN2_VL_MODEL_CONFIG
from .qwen2_vl_layers import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
    VisionAttention,
    VisionMLP,
)

# ----------------------------------------------------------------------------
# LLM
# ----------------------------------------------------------------------------


def qwen2_decoder_block(
    x,
    cos,
    sin,
    causal_mask,
    hidden_size,
    num_heads,
    num_kv_heads,
    intermediate_size,
    rms_eps,
    layer_idx,
):
    prefix = f"model_layers_{layer_idx}"
    ln1 = Qwen2RMSNorm(hidden_size, eps=rms_eps, name=f"{prefix}_input_layernorm")(x)
    attn_out = Qwen2Attention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        name_prefix=f"{prefix}_self_attn",
    )(ln1, cos=cos, sin=sin, attention_mask=causal_mask)
    x = layers.Add()([x, attn_out])

    ln2 = Qwen2RMSNorm(
        hidden_size, eps=rms_eps, name=f"{prefix}_post_attention_layernorm"
    )(x)
    mlp_out = Qwen2MLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        name_prefix=f"{prefix}_mlp",
    )(ln2)
    x = layers.Add()([x, mlp_out])
    return x


def build_qwen2_llm(cfg, name="qwen2_llm"):
    tc = cfg["text_config"]
    hidden = tc["hidden_size"]
    n_layers = tc["num_hidden_layers"]
    n_heads = tc["num_attention_heads"]
    n_kv = tc["num_key_value_heads"]
    inter = tc["intermediate_size"]
    rms_eps = tc["rms_norm_eps"]

    # Inputs: embedded token states + pre-built cos/sin + additive causal mask
    inp_embed = layers.Input(shape=(None, hidden), name="inputs_embeds")
    cos = layers.Input(shape=(1, None, hidden // n_heads), name="rope_cos")
    sin = layers.Input(shape=(1, None, hidden // n_heads), name="rope_sin")
    causal_mask = layers.Input(shape=(1, None, None), name="causal_mask")

    x = inp_embed
    for i in range(n_layers):
        x = qwen2_decoder_block(
            x,
            cos,
            sin,
            causal_mask,
            hidden_size=hidden,
            num_heads=n_heads,
            num_kv_heads=n_kv,
            intermediate_size=inter,
            rms_eps=rms_eps,
            layer_idx=i,
        )
    x = Qwen2RMSNorm(hidden, eps=rms_eps, name="model_norm")(x)

    return keras.Model(
        inputs={
            "inputs_embeds": inp_embed,
            "rope_cos": cos,
            "rope_sin": sin,
            "causal_mask": causal_mask,
        },
        outputs=x,
        name=name,
    )


# ----------------------------------------------------------------------------
# Vision tower
# ----------------------------------------------------------------------------


def vision_block(x, cos, sin, embed_dim, num_heads, intermediate_size, layer_idx):
    prefix = f"visual_blocks_{layer_idx}"
    ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_norm1")(x)
    attn = VisionAttention(
        hidden_size=embed_dim,
        num_heads=num_heads,
        name_prefix=f"{prefix}_attn",
    )(ln1, cos=cos, sin=sin)
    x = layers.Add()([x, attn])
    ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_norm2")(x)
    mlp = VisionMLP(
        hidden_size=embed_dim,
        intermediate_size=intermediate_size,
        name_prefix=f"{prefix}_mlp",
    )(ln2)
    x = layers.Add()([x, mlp])
    return x


def build_qwen2_vision(cfg, name="qwen2_vision"):
    vc = cfg["vision_config"]
    embed_dim = vc["embed_dim"]
    hidden_size = vc["hidden_size"]  # LLM hidden, merger target
    depth = vc["depth"]
    num_heads = vc["num_heads"]
    mlp_ratio = vc["mlp_ratio"]
    patch_size = vc["patch_size"]
    temporal_patch_size = vc["temporal_patch_size"]
    merge_size = vc["spatial_merge_size"]
    intermediate = embed_dim * mlp_ratio

    # Inputs:
    #   pixel_values: (N_patches, C * temporal * H * W) where each row is one
    #     flattened 3×2×14×14 = 1176-dim patch.
    pixel_values = layers.Input(
        shape=(3 * temporal_patch_size * patch_size * patch_size,),
        name="pixel_values",
    )
    rotary_cos = layers.Input(shape=(embed_dim // num_heads,), name="vision_cos")
    rotary_sin = layers.Input(shape=(embed_dim // num_heads,), name="vision_sin")

    # Patch embed: 3D conv (C=3, T=2, H=14, W=14) → embed_dim.
    # Implement as a Dense on flattened patch (pre-flattened outside model).
    x = layers.Dense(embed_dim, use_bias=False, name="visual_patch_embed_proj")(
        pixel_values
    )  # (N, embed_dim)
    # Add a pseudo-batch axis so attention ops work.
    x = layers.Lambda(
        lambda t: ops.expand_dims(t, 0),
        output_shape=lambda s: (1,) + tuple(s),
        name="vision_add_batch",
    )(x)

    for i in range(depth):
        x = vision_block(
            x,
            rotary_cos,
            rotary_sin,
            embed_dim=embed_dim,
            num_heads=num_heads,
            intermediate_size=intermediate,
            layer_idx=i,
        )

    # Merger: LN + group (merge_size**2 patches) + MLP to LLM hidden size.
    x = layers.Lambda(
        lambda t: t[0],
        output_shape=lambda s: (s[1], s[2]),
        name="vision_drop_batch",
    )(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="visual_merger_ln_q")(x)
    # Group every (merge_size**2) consecutive patches along N axis.
    x = layers.Lambda(
        lambda t, g=merge_size**2: ops.reshape(t, (-1, g * embed_dim)),
        output_shape=(merge_size**2 * embed_dim,),
        name="visual_merger_group",
    )(x)
    x = layers.Dense(
        embed_dim * merge_size * merge_size,
        use_bias=True,
        name="visual_merger_mlp_0",
    )(x)
    x = layers.Lambda(
        lambda t: t * ops.sigmoid(1.702 * t),
        output_shape=lambda s: s,
        name="visual_merger_act",
    )(x)
    x = layers.Dense(hidden_size, use_bias=True, name="visual_merger_mlp_2")(x)

    return keras.Model(
        inputs={
            "pixel_values": pixel_values,
            "vision_cos": rotary_cos,
            "vision_sin": rotary_sin,
        },
        outputs=x,
        name=name,
    )


# ----------------------------------------------------------------------------
# Utility builders
# ----------------------------------------------------------------------------


def build_llm_inv_freq(head_dim: int, rope_theta: float):
    half = head_dim // 2
    idx = np.arange(0, half, dtype=np.float32)
    return 1.0 / (rope_theta ** (idx / half))


def build_vision_inv_freq(head_dim: int, rope_theta: float = 10000.0):
    # Vision uses head_dim/4 frequencies (half for h, half for w → head_dim/2
    # total rotation dims, split-RoPE duplicated to head_dim).
    quarter = head_dim // 4
    idx = np.arange(0, quarter, dtype=np.float32)
    return 1.0 / (rope_theta ** (idx / quarter))


def make_causal_mask(seq_len):
    i = ops.arange(seq_len)[:, None]
    j = ops.arange(seq_len)[None, :]
    mask = ops.cast(j > i, "float32") * -1e9
    return mask[None, None, :, :]


def make_text_position_ids(seq_len, batch=1):
    """For text-only input, all three axes of M-RoPE share 0..T-1."""
    pos = ops.arange(seq_len)[None, :]
    pos = ops.broadcast_to(pos, (batch, seq_len))
    return ops.stack([pos, pos, pos], axis=0)  # (3, B, T)


# ----------------------------------------------------------------------------
# Top-level factories
# ----------------------------------------------------------------------------


def _build_qwen2_vl(variant_key: str, weights, name: str) -> dict:
    """Build the bundle for ``variant_key`` and optionally hydrate from HF.

    Args:
        variant_key: A key in :data:`QWEN2_VL_MODEL_CONFIG`.
        weights: ``"qwen"`` to download + convert + cache the official HF
            checkpoint on first use; ``None`` for random init; or a path
            to a pre-converted directory.
        name: Name prefix for the Keras sub-models.

    Returns:
        ``{config, embed_tokens, llm, vision, lm_head?, llm_inv_freq,
        vision_inv_freq}``. ``lm_head`` is present iff
        ``tie_word_embeddings=False`` for that variant.
    """
    cfg = QWEN2_VL_MODEL_CONFIG[variant_key]
    tc = cfg["text_config"]
    vc = cfg["vision_config"]

    slug = name.lower()
    embed_tokens = keras.layers.Embedding(
        input_dim=tc["vocab_size"],
        output_dim=tc["hidden_size"],
        name="model_embed_tokens",
    )
    llm = build_qwen2_llm(cfg, name=f"{slug}_llm")
    vision = build_qwen2_vision(cfg, name=f"{slug}_vision")

    head_dim = tc["hidden_size"] // tc["num_attention_heads"]
    llm_inv_freq = build_llm_inv_freq(head_dim, tc["rope_theta"])
    vision_head_dim = vc["embed_dim"] // vc["num_heads"]
    vision_inv_freq = build_vision_inv_freq(vision_head_dim)

    bundle = {
        "config": cfg,
        "embed_tokens": embed_tokens,
        "llm": llm,
        "vision": vision,
        "llm_inv_freq": llm_inv_freq,
        "vision_inv_freq": vision_inv_freq,
    }

    # Untied variants (7B / 72B) need a separate output projection.
    if not tc["tie_word_embeddings"]:
        bundle["lm_head"] = keras.layers.Dense(
            tc["vocab_size"], use_bias=False, name="lm_head"
        )

    if weights == "qwen":
        from .convert_qwen2_vl_hf_to_keras import (
            load_and_convert_bundle_from_hf,
            transfer_qwen2_vl_weights,
        )

        load_and_convert_bundle_from_hf(
            bundle=bundle,
            model_name=slug,
            hf_model_id=QWEN2_VL_HF_CONVERT_VARIANTS[variant_key],
            transfer_fn=transfer_qwen2_vl_weights,
        )
    elif weights is not None:
        raise ValueError(
            f"Unknown weights preset {weights!r}. Expected 'qwen' or None."
        )
    else:
        print("No weights loaded.")

    return bundle


@register_model
def Qwen2VL2B(weights="qwen", name="Qwen2VL2B"):
    """Qwen2-VL 2B base — 1.5 B params, 28 layers, ``hidden=1536``, tied LM head.

    Loads from ``Qwen/Qwen2-VL-2B`` on HF.
    """
    return _build_qwen2_vl("Qwen2VL2B", weights, name)


@register_model
def Qwen2VL2BInstruct(weights="qwen", name="Qwen2VL2BInstruct"):
    """Qwen2-VL 2B Instruct — same architecture as 2B, instruction-tuned weights.

    Loads from ``Qwen/Qwen2-VL-2B-Instruct`` on HF.
    """
    return _build_qwen2_vl("Qwen2VL2BInstruct", weights, name)


@register_model
def Qwen2VL7B(weights="qwen", name="Qwen2VL7B"):
    """Qwen2-VL 7B base — 8 B params, 28 layers, ``hidden=3584``, untied LM head.

    Loads from ``Qwen/Qwen2-VL-7B`` on HF.
    """
    return _build_qwen2_vl("Qwen2VL7B", weights, name)


@register_model
def Qwen2VL7BInstruct(weights="qwen", name="Qwen2VL7BInstruct"):
    """Qwen2-VL 7B Instruct — same architecture as 7B, instruction-tuned weights.

    Loads from ``Qwen/Qwen2-VL-7B-Instruct`` on HF.
    """
    return _build_qwen2_vl("Qwen2VL7BInstruct", weights, name)


@register_model
def Qwen2VL72B(weights="qwen", name="Qwen2VL72B"):
    """Qwen2-VL 72B base — 72 B params, 80 layers, ``hidden=8192``, untied LM head.

    Loads from ``Qwen/Qwen2-VL-72B`` on HF. Requires ~150 GB RAM during
    conversion.
    """
    return _build_qwen2_vl("Qwen2VL72B", weights, name)


@register_model
def Qwen2VL72BInstruct(weights="qwen", name="Qwen2VL72BInstruct"):
    """Qwen2-VL 72B Instruct — same architecture as 72B, instruction-tuned weights.

    Loads from ``Qwen/Qwen2-VL-72B-Instruct`` on HF. Requires ~150 GB RAM
    during conversion.
    """
    return _build_qwen2_vl("Qwen2VL72BInstruct", weights, name)
