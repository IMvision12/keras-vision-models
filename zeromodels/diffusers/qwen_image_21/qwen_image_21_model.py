from __future__ import annotations

import keras
import numpy as np
from keras import layers, ops

from zeromodels.base import BaseModel, CausalMask
from zeromodels.diffusers.qwen_image.qwen_image_model import QwenImageFamily
from zeromodels.diffusers.qwen_image_21.qwen_image_21_config import (
    DEFAULT_LATENTS_MEAN,
    DEFAULT_LATENTS_STD,
    QwenImage21Config,
    QwenImage21TextConfig,
    QwenImage21TransformerConfig,
    QwenImage21VAEConfig,
)
from zeromodels.diffusers.qwen_image_21.qwen_image_21_layers import (
    IMG_TOKENS_PER_SLOT,
    MASK_NEG,
    QwenImage21AdaLayerNormContinuous,
    QwenImage21TextProjection,
    QwenImage21TimestepProjEmbeddings,
    QwenImage21TransformerBlock,
    build_block_causal_additive_mask,
    build_qwenimage21_rope_angles,
    build_token_metadata,
)
from zeromodels.diffusers.qwen_image_21.qwen_image_21_vae import (
    QwenImage21CausalConv,
    QwenImage21Decoder3d,
    QwenImage21Encoder3d,
)
from zeromodels.diffusers.stable_diffusion.stable_diffusion_layers import safe_name
from zeromodels.models.qwen3_vl.qwen3_vl_model import (
    Qwen3VLTextModel,
    qwen3_text_cos_sin,
)


def pack_latents(latents, height, width):
    """Flatten ``(B, H, W, C)`` → ``(B, H*W, C)`` (2.1 consumes latents unpatched)."""
    batch = ops.shape(latents)[0]
    channels = ops.shape(latents)[-1]
    return ops.reshape(latents, (batch, height * width, channels))


def unpack_latents(latents, height, width, channels):
    """Unflatten ``(B, H*W, C)`` → ``(B, H, W, C)``."""
    batch = ops.shape(latents)[0]
    return ops.reshape(latents, (batch, height, width, channels))


@keras.saving.register_keras_serializable(package="zeromodels")
class AutoencoderKLQwenImage21(BaseModel):
    """Qwen-Image-2.1 VAE (Diffusers ``AutoencoderKLQwenImage21``), channels-last.

    64-channel latents, 16× spatial compression, residual encoder/decoder, RGBA
    (4-channel) IO. Graph built for ``sample_size``; weights are resolution-
    independent.
    """

    config_class = QwenImage21VAEConfig
    HF_MODEL_TYPE = None

    def __init__(
        self,
        base_dim=96,
        decoder_base_dim=144,
        z_dim=64,
        dim_mult=(1, 2, 4, 8, 8),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(False, True, True, True),
        dropout=0.0,
        input_channels=4,
        out_channels=4,
        is_residual=True,
        scale_factor_spatial=16,
        latents_mean=DEFAULT_LATENTS_MEAN,
        latents_std=DEFAULT_LATENTS_STD,
        sample_size=1024,
        name="AutoencoderKLQwenImage21",
        **kwargs,
    ):
        dim_mult = tuple(dim_mult)
        temperal_downsample = tuple(temperal_downsample)
        attn_scales = tuple(attn_scales)
        latents_mean = tuple(latents_mean)
        latents_std = tuple(latents_std)
        temperal_upsample = tuple(reversed(temperal_downsample))
        decoder_base_dim = (
            int(decoder_base_dim) if decoder_base_dim is not None else int(base_dim)
        )

        h_img, w_img = (
            sample_size
            if isinstance(sample_size, (tuple, list))
            else (sample_size, sample_size)
        )
        spatial_compression_ratio = int(scale_factor_spatial)
        h_lat, w_lat = (
            h_img // spatial_compression_ratio,
            w_img // spatial_compression_ratio,
        )

        encoder = QwenImage21Encoder3d(
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            input_channels=input_channels,
            is_residual=is_residual,
            module_path="encoder",
        )
        decoder = QwenImage21Decoder3d(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=temperal_upsample,
            dropout=dropout,
            out_channels=out_channels,
            is_residual=is_residual,
            module_path="decoder",
        )
        quant_conv = QwenImage21CausalConv(
            z_dim * 2, kernel_size=1, padding=0, module_path="quant_conv"
        )
        post_quant_conv = QwenImage21CausalConv(
            z_dim, kernel_size=1, padding=0, module_path="post_quant_conv"
        )

        image_in = layers.Input(shape=(h_img, w_img, input_channels), name="image")
        latent_in = layers.Input(shape=(h_lat, w_lat, z_dim), name="latent")

        image_5d = ops.expand_dims(image_in, axis=1)
        moments_5d = quant_conv(encoder(image_5d))
        moments = ops.squeeze(moments_5d, axis=1)

        latent_5d = ops.expand_dims(latent_in, axis=1)
        decoded_5d = decoder(post_quant_conv(latent_5d))
        decoded = ops.squeeze(decoded_5d, axis=1)
        decoded = ops.clip(decoded, -1.0, 1.0)

        super().__init__(
            inputs={"image": image_in, "latent": latent_in},
            outputs={"moments": moments, "sample": decoded},
            name=name,
            **kwargs,
        )

        self.base_dim = base_dim
        self.decoder_base_dim = decoder_base_dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_upsample
        self.dropout = dropout
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.is_residual = is_residual
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        self.sample_size = sample_size
        self.spatial_compression_ratio = spatial_compression_ratio
        self.vae_scale_factor = spatial_compression_ratio
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv

    def to_ndhwc(self, x, is_latent=False):
        static_ndim = len(x.shape)
        if static_ndim == 4:
            return ops.expand_dims(x, axis=1)
        if static_ndim == 5:
            c1 = int(x.shape[1]) if x.shape[1] is not None else None
            c_last = int(x.shape[-1]) if x.shape[-1] is not None else None
            expect = self.z_dim if is_latent else self.input_channels
            if c1 == expect and c_last != expect:
                return ops.transpose(x, (0, 2, 3, 4, 1))
            return x
        raise ValueError(f"Expected 4D or 5D tensor, got shape {x.shape}")

    def encode(self, x, sample=False, seed=None):
        was_4d = len(x.shape) == 4
        x5 = self.to_ndhwc(x, is_latent=False)
        moments = self.quant_conv(self.encoder(x5))
        mean, logvar = ops.split(moments, 2, axis=-1)
        if sample:
            logvar = ops.clip(logvar, -30.0, 20.0)
            std = ops.exp(0.5 * logvar)
            noise = keras.random.normal(ops.shape(mean), dtype=mean.dtype, seed=seed)
            z = mean + std * noise
        else:
            z = mean
        if was_4d:
            z = ops.squeeze(z, axis=1)
        return z

    def decode(self, z):
        was_4d = len(z.shape) == 4
        z5 = self.to_ndhwc(z, is_latent=True)
        out = self.decoder(self.post_quant_conv(z5))
        out = ops.clip(out, -1.0, 1.0)
        if was_4d:
            out = ops.squeeze(out, axis=1)
        return out

    @classmethod
    def kwargs_from_diffusers_config(cls, cfg):
        return {
            "base_dim": cfg.get("base_dim", 96),
            "decoder_base_dim": cfg.get("decoder_base_dim", 144),
            "z_dim": cfg.get("z_dim", 64),
            "dim_mult": tuple(cfg.get("dim_mult", (1, 2, 4, 8, 8))),
            "num_res_blocks": cfg.get("num_res_blocks", 2),
            "attn_scales": tuple(cfg.get("attn_scales") or ()),
            "temperal_downsample": tuple(
                cfg.get("temperal_downsample", (False, True, True, True))
            ),
            "dropout": cfg.get("dropout", 0.0),
            "input_channels": cfg.get("in_channels", 4),
            "out_channels": cfg.get("out_channels", 4),
            "is_residual": bool(cfg.get("is_residual", True)),
            "scale_factor_spatial": cfg.get("scale_factor_spatial", 16),
            "latents_mean": tuple(cfg.get("latents_mean", DEFAULT_LATENTS_MEAN)),
            "latents_std": tuple(cfg.get("latents_std", DEFAULT_LATENTS_STD)),
        }


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21Transformer2DModel(BaseModel):
    """Qwen-Image-2.1 denoiser (Diffusers ``QwenImage21Transformer2DModel``).

    Single-stream block-causal DiT over unpatched latents and Qwen3-VL text
    features. Built for a fixed T2I layout (``sample_size`` latent side +
    ``text_seq_len``). Output ``{"sample": target velocity}``.
    """

    HF_MODEL_TYPE = None
    config_class = QwenImage21TransformerConfig

    def __init__(
        self,
        patch_size=1,
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
        axes_dims_rope=(16, 56, 56),
        eps=1e-6,
        causal_condition=True,
        sample_size=64,
        text_seq_len=512,
        name="QwenImage21Transformer2DModel",
        **kwargs,
    ):
        keras_kwargs = {k: kwargs.pop(k) for k in ("trainable", "dtype") if k in kwargs}
        axes_dims_rope = tuple(axes_dims_rope)
        inner_dim = num_attention_heads * attention_head_dim
        sample_h = (
            sample_size[0] if isinstance(sample_size, (tuple, list)) else sample_size
        )
        sample_w = (
            sample_size[1] if isinstance(sample_size, (tuple, list)) else sample_size
        )
        assert patch_size == 1, "Qwen-Image-2.1 consumes latents unpatched"
        img_seq = sample_h * sample_w
        img_shapes = [(1, sample_h, sample_w)]

        target_slots = (sample_h * sample_w) // IMG_TOKENS_PER_SLOT
        slot_mask = np.concatenate(
            [
                np.zeros(text_seq_len, dtype=bool),
                np.ones(target_slots, dtype=bool),
            ]
        )
        image_pad_mask = np.repeat(
            slot_mask, np.where(slot_mask, IMG_TOKENS_PER_SLOT, 1)
        )
        joint_seq = int(image_pad_mask.shape[0])
        image_ids, target_token_mask = build_token_metadata(image_pad_mask, img_shapes)
        rope_angles = build_qwenimage21_rope_angles(
            img_shapes, image_pad_mask, axes_dims_rope
        )
        attn_mask_np = build_block_causal_additive_mask(image_ids)

        time_text_embed = QwenImage21TimestepProjEmbeddings(
            embedding_dim=inner_dim, module_path="time_text_embed"
        )
        txt_in = QwenImage21TextProjection(
            context_in_dim, inner_dim, eps=eps, module_path="txt_in"
        )
        img_in = layers.Dense(inner_dim, use_bias=False, name=safe_name("img_in"))
        modulation = layers.Dense(
            4 * inner_dim, use_bias=False, name=safe_name("modulation.1")
        )
        blocks = [
            QwenImage21TransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                eps=eps,
                module_path=f"transformer_blocks.{i}",
            )
            for i in range(num_layers)
        ]
        norm_out = QwenImage21AdaLayerNormContinuous(
            inner_dim, inner_dim, eps=eps, module_path="norm_out"
        )
        proj_out = layers.Dense(
            patch_size * patch_size * (out_channels or in_channels),
            use_bias=False,
            name=safe_name("proj_out"),
        )

        sample_in = layers.Input(shape=(img_seq, in_channels), name="sample")
        timestep_in = layers.Input(shape=(), name="timestep")
        enc_in = layers.Input(
            shape=(text_seq_len, context_in_dim), name="encoder_hidden_states"
        )
        enc_mask_in = layers.Input(
            shape=(text_seq_len,), dtype="int32", name="encoder_hidden_states_mask"
        )

        hidden = img_in(sample_in)
        encoder = txt_in(enc_in)

        joint = ops.concatenate([encoder, hidden], axis=1)

        # The graph reserves ``text_seq_len`` slots, but Diffusers places image
        # tokens immediately after the actual (batch-padded) prompt. Only the
        # frame axis of image RoPE depends on that starting position.
        frame_dims = axes_dims_rope[0] // 2
        frame_freqs = np.power(
            10000.0,
            -np.arange(0, axes_dims_rope[0], 2, dtype=np.float64) / axes_dims_rope[0],
        ).astype(np.float32)
        prompt_len = ops.max(ops.sum(enc_mask_in, axis=1))
        frame_angles = ops.cast(prompt_len, "float32") * ops.convert_to_tensor(
            frame_freqs, dtype="float32"
        )
        image_angles = ops.concatenate(
            [
                ops.broadcast_to(frame_angles, (img_seq, frame_dims)),
                ops.convert_to_tensor(
                    rope_angles[text_seq_len:, frame_dims:], dtype="float32"
                ),
            ],
            axis=-1,
        )
        rotary = ops.concatenate(
            [
                ops.convert_to_tensor(rope_angles[:text_seq_len], dtype="float32"),
                image_angles,
            ],
            axis=0,
        )
        attn_mask = ops.convert_to_tensor(attn_mask_np)
        target_mask_t = ops.convert_to_tensor(target_token_mask)
        text_pos = np.flatnonzero(~image_pad_mask).astype(np.int32)

        text_scatter = np.zeros((joint_seq, text_seq_len), dtype=np.float32)
        for j, pos in enumerate(text_pos):
            text_scatter[pos, j] = 1.0
        text_scatter_t = ops.convert_to_tensor(text_scatter)
        text_valid_on_joint = ops.einsum(
            "bt,jt->bj", ops.cast(enc_mask_in, "float32"), text_scatter_t
        )
        is_text = ops.convert_to_tensor((~image_pad_mask).astype(np.float32))
        key_ok = text_valid_on_joint * is_text[None, :] + (1.0 - is_text[None, :])
        attn_mask = ops.add(attn_mask, (1.0 - key_ok)[:, None, None, :] * MASK_NEG)

        if causal_condition:
            t_all = ops.concatenate(
                [timestep_in, ops.zeros((1,), dtype=timestep_in.dtype)], axis=0
            )
            temb = time_text_embed(t_all)
            mod_mask = target_mask_t
        else:
            temb = time_text_embed(timestep_in)
            mod_mask = None

        mod = modulation(ops.silu(temb))
        for block in blocks:
            joint = block(
                joint,
                mod,
                rotary_emb=rotary,
                attention_mask=attn_mask,
                target_token_mask=mod_mask,
            )
        joint = norm_out([joint, temb, mod_mask])
        output_full = proj_out(joint)
        output = output_full[:, -img_seq:, :]
        super().__init__(
            inputs={
                "sample": sample_in,
                "timestep": timestep_in,
                "encoder_hidden_states": enc_in,
                "encoder_hidden_states_mask": enc_mask_in,
            },
            outputs={"sample": output},
            name=name,
            **keras_kwargs,
        )

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.context_in_dim = context_in_dim
        self.mlp_ratio = mlp_ratio
        self.axes_dims_rope = axes_dims_rope
        self.eps = eps
        self.causal_condition = causal_condition
        self.sample_size = sample_size
        self.text_seq_len = text_seq_len
        self.inner_dim = inner_dim
        self.img_seq = img_seq
        self.joint_seq = joint_seq
        self.time_text_embed = time_text_embed
        self.txt_in = txt_in
        self.img_in = img_in
        self.modulation = modulation
        self.transformer_blocks = blocks
        self.norm_out = norm_out
        self.proj_out = proj_out

    @classmethod
    def kwargs_from_diffusers_config(cls, cfg):
        return {
            "patch_size": cfg.get("patch_size", 1),
            "in_channels": cfg.get("in_channels", 64),
            "out_channels": cfg.get("out_channels", 64),
            "num_layers": cfg.get("num_layers", 32),
            "attention_head_dim": cfg.get("attention_head_dim", 128),
            "num_attention_heads": cfg.get("num_attention_heads", 32),
            "context_in_dim": cfg.get("context_in_dim", 4096),
            "mlp_ratio": cfg.get("mlp_ratio", 3),
            "axes_dims_rope": tuple(cfg.get("axes_dims_rope", (16, 56, 56))),
            "eps": cfg.get("eps", 1e-6),
            "causal_condition": bool(cfg.get("causal_condition", True)),
        }


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21TextEncoderModel(BaseModel):
    """Qwen-Image-2.1 prompt encoder: Qwen3-VL text tower, no vision / LM head.

    Returns **pre-final-norm** hidden states (Diffusers hooks the last RMSNorm so
    the transformer sees the decoder output before it).
    """

    HF_MODEL_TYPE = None
    config_class = QwenImage21TextConfig
    output_logits = False

    def __init__(self, name="text_encoder", **kwargs):
        keras_kwargs = {k: kwargs.pop(k) for k in ("trainable", "dtype") if k in kwargs}
        if "max_seq_len" in kwargs and not any(
            k in kwargs for k in ("embed_dim", "num_layers", "vocab_size")
        ):
            pass
        config = self.config_class.from_dict(kwargs) if kwargs else self.config_class()
        max_seq_len = int(getattr(config, "max_seq_len", 1024))

        language_model = Qwen3VLTextModel(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            mlp_dim=config.mlp_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            norm_eps=config.norm_eps,
            name="language_model",
        )
        causal_mask = CausalMask(name="causal_mask")

        input_ids = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        attention_mask = layers.Input(
            shape=(None,), dtype="int32", name="attention_mask"
        )
        hidden = language_model.token_embedding(input_ids)
        pos1 = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
        pos = ops.stack([pos1, pos1, pos1], axis=0)
        cos, sin = qwen3_text_cos_sin(
            pos, config.head_dim, config.rope_theta, tuple(config.mrope_section)
        )
        mask = causal_mask(input_ids, attention_mask)
        for layer in language_model.decoder_layers:
            hidden = layer(hidden, cos, sin, attention_mask=mask)

        super().__init__(
            inputs={"input_ids": input_ids, "attention_mask": attention_mask},
            outputs={"last_hidden_state": hidden},
            name=name,
            **keras_kwargs,
        )
        self.language_model = language_model
        self.causal_mask_layer = causal_mask
        self.max_seq_len = max_seq_len
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.mlp_dim = config.mlp_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.norm_eps = config.norm_eps
        self.rope_theta = config.rope_theta
        self.mrope_section = tuple(config.mrope_section)
        self.tie_embeddings = config.tie_embeddings


class QwenImage21Family(QwenImageFamily):
    """Qwen-Image-2.1 inside :class:`ZMDiffusion` (``model_type="qwen_image_21"``).

    Three components, ``transformer`` (the single-stream DiT,
    :class:`QwenImage21Transformer2DModel`), ``vae`` (the residual RGBA
    :class:`AutoencoderKLQwenImage21`) and ``text_encoder`` (the Qwen3-VL text tower,
    pre-final-norm states), wired like Qwen-Image's (diffusers'
    ``QwenImage21Pipeline`` components). Hosted as ``zeromodels/qwen-image-2.1``.

    Generation is Qwen-Image's with the 2.1 differences: unpatched latents (one
    token per latent pixel), a plain guidance combination, the alpha channel dropped
    from the decoded image, and the reference defaults of 40 steps without guidance.
    Classifier-free guidance runs only with ``guidance_scale > 1`` and an explicit
    negative prompt.
    """

    model_type = "qwen_image_21"
    config_class = QwenImage21Config
    generate_args = {"num_inference_steps": 40, "guidance_scale": 1.0}

    def build_components(self, config, data_format, channels_axis):
        d, v = config.transformer_config, config.vae_config
        transformer = QwenImage21Transformer2DModel(
            patch_size=d.patch_size,
            in_channels=d.in_channels,
            out_channels=d.out_channels,
            num_layers=d.num_layers,
            attention_head_dim=d.attention_head_dim,
            num_attention_heads=d.num_attention_heads,
            context_in_dim=d.context_in_dim,
            mlp_ratio=d.mlp_ratio,
            axes_dims_rope=d.axes_dims_rope,
            eps=d.eps,
            causal_condition=d.causal_condition,
            sample_size=d.sample_size,
            text_seq_len=config.max_sequence_length,
        )
        vae = AutoencoderKLQwenImage21(
            base_dim=v.base_dim,
            decoder_base_dim=v.decoder_base_dim,
            z_dim=v.z_dim,
            dim_mult=v.dim_mult,
            num_res_blocks=v.num_res_blocks,
            attn_scales=v.attn_scales,
            temperal_downsample=v.temperal_downsample,
            dropout=v.dropout,
            input_channels=v.input_channels,
            out_channels=v.out_channels,
            is_residual=v.is_residual,
            scale_factor_spatial=v.scale_factor_spatial,
            latents_mean=v.latents_mean,
            latents_std=v.latents_std,
            sample_size=v.sample_size,
        )
        text_kw = (
            config.text_config.constructor_kwargs()
            if hasattr(config.text_config, "constructor_kwargs")
            else dict(config.text_config)
        )
        text_encoder = QwenImage21TextEncoderModel(**text_kw)
        return {"transformer": transformer, "vae": vae, "text_encoder": text_encoder}

    def build_graph(self, config, components, data_format):
        d, v, t = config.transformer_config, config.vae_config, config.text_config
        transformer, vae, text_encoder = (
            components["transformer"],
            components["vae"],
            components["text_encoder"],
        )
        text_seq = config.max_sequence_length
        img_h = img_w = (
            v.sample_size
            if not isinstance(v.sample_size, (tuple, list))
            else v.sample_size[0]
        )
        lat_h = img_h // vae.vae_scale_factor
        lat_w = img_w // vae.vae_scale_factor

        inputs = {
            "sample": layers.Input(
                shape=(transformer.img_seq, d.in_channels), name="sample"
            ),
            "timestep": layers.Input(shape=(), name="timestep"),
            "encoder_hidden_states": layers.Input(
                shape=(text_seq, d.context_in_dim), name="encoder_hidden_states"
            ),
            "encoder_hidden_states_mask": layers.Input(
                shape=(text_seq,), dtype="int32", name="encoder_hidden_states_mask"
            ),
            "image": layers.Input(shape=(img_h, img_w, v.input_channels), name="image"),
            "latent": layers.Input(shape=(lat_h, lat_w, v.z_dim), name="latent"),
            "token_ids": layers.Input(
                shape=(t.max_seq_len,), dtype="int32", name="token_ids"
            ),
            "padding_mask": layers.Input(
                shape=(t.max_seq_len,), dtype="int32", name="padding_mask"
            ),
        }
        noise_pred = transformer(
            {
                "sample": inputs["sample"],
                "timestep": inputs["timestep"],
                "encoder_hidden_states": inputs["encoder_hidden_states"],
                "encoder_hidden_states_mask": inputs["encoder_hidden_states_mask"],
            }
        )["sample"]
        vae_out = vae({"image": inputs["image"], "latent": inputs["latent"]})
        text_out = text_encoder(
            {"input_ids": inputs["token_ids"], "attention_mask": inputs["padding_mask"]}
        )
        return inputs, {
            "noise_pred": noise_pred,
            "moments": vae_out["moments"],
            "image": vae_out["sample"],
            "prompt_embeds": text_out["last_hidden_state"],
        }

    def image_seq_len(self, height, width):
        return height * width

    def latent_shape(self, model):
        h, w = self.latent_side(model)
        return (self.image_seq_len(h, w), model.vae.z_dim)

    def pack_latents(self, latents, height, width):
        return pack_latents(latents, height, width)

    def unpack_latents(self, latents, height, width, channels):
        return unpack_latents(latents, height, width, channels)

    def unconditional_ids(self, model, batch):
        length = model.config.text_config.max_seq_len
        row = [model.config.pad_token_id] * length
        return ops.convert_to_tensor([row] * batch, dtype="int32")

    def decode_latents(self, model, latents, height=None, width=None):
        image = super().decode_latents(model, latents, height=height, width=width)
        if int(image.shape[-1]) == 4:
            image = image[..., :3]  # RGBA VAE: drop the alpha channel
        return image

    def combine_guidance(self, noise_pred, neg_pred, guidance_scale):
        return neg_pred + guidance_scale * (noise_pred - neg_pred)

    def negative_embeddings(
        self,
        model,
        batch,
        negative_input_ids,
        negative_attention_mask,
        guidance_scale,
        conditioning,
    ):
        # guidance only with an explicit negative prompt (the reference's default
        # is no guidance at all)
        if guidance_scale <= 1.0 or negative_input_ids is None:
            return None
        return model.encode_prompt(negative_input_ids, negative_attention_mask)
