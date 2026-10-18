from __future__ import annotations

import keras
import numpy as np
from keras import layers, ops

from zeromodels.base import BaseModel
from zeromodels.base.base_attention import use_attn_implementation
from zeromodels.base.base_mixin import inference_scope
from zeromodels.base.base_scheduler import FlowMatchEulerDiscreteScheduler
from zeromodels.diffusers.qwen_image.qwen_image_config import (
    DEFAULT_LATENTS_MEAN,
    DEFAULT_LATENTS_STD,
    QwenImageConfig,
    QwenImageTextConfig,
    QwenImageTransformerConfig,
    QwenImageVAEConfig,
)
from zeromodels.diffusers.qwen_image.qwen_image_layers import (
    QwenImageAdaLayerNormContinuous,
    QwenImageEmbedRope,
    QwenImageRMSNorm,
    QwenImageTimestepProjEmbeddings,
    QwenImageTransformerBlock,
)
from zeromodels.diffusers.qwen_image.qwen_image_vae import (
    QwenImageCausalConv3d,
    QwenImageDecoder3d,
    QwenImageEncoder3d,
)
from zeromodels.diffusers.stable_diffusion.stable_diffusion_layers import safe_name
from zeromodels.models.qwen2_5_vl.qwen2_5_vl_model import Qwen2_5VLModel


def pack_latents(latents, height, width):
    """Pack ``(B, H, W, C)`` latents into ``(B, H/2 * W/2, C*4)`` (Diffusers)."""
    batch = ops.shape(latents)[0]
    channels = ops.shape(latents)[-1]
    latents = ops.reshape(latents, (batch, height // 2, 2, width // 2, 2, channels))
    latents = ops.transpose(latents, (0, 1, 3, 5, 2, 4))
    return ops.reshape(latents, (batch, (height // 2) * (width // 2), channels * 4))


def unpack_latents(latents, height, width, channels):
    """Unpack ``(B, seq, C*4)`` to ``(B, H, W, C)``."""
    batch = ops.shape(latents)[0]
    latents = ops.reshape(latents, (batch, height // 2, width // 2, channels, 2, 2))
    latents = ops.transpose(latents, (0, 1, 4, 2, 5, 3))
    return ops.reshape(latents, (batch, height, width, channels))


@keras.saving.register_keras_serializable(package="zeromodels")
class AutoencoderKLQwenImage(BaseModel):
    """Qwen-Image VAE (Diffusers ``AutoencoderKLQwenImage``), channels-last NDHWC.

    Encode / decode a single image frame (``T=1``). The functional graph is built
    for ``sample_size`` but conv weights are resolution-independent. Public
    helpers accept channels-last HWC images or Diffusers NCDHW 5D tensors.

    Latent normalisation with ``latents_mean`` / ``latents_std`` is left to the
    pipeline (stored on the model for that purpose).
    """

    config_class = QwenImageVAEConfig
    HF_MODEL_TYPE = None

    def __init__(
        self,
        base_dim=96,
        z_dim=16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(False, True, True),
        dropout=0.0,
        input_channels=3,
        latents_mean=DEFAULT_LATENTS_MEAN,
        latents_std=DEFAULT_LATENTS_STD,
        sample_size=1024,
        apply_temporal=False,
        name="AutoencoderKLQwenImage",
        **kwargs,
    ):
        dim_mult = tuple(dim_mult)
        temperal_downsample = tuple(temperal_downsample)
        attn_scales = tuple(attn_scales)
        latents_mean = tuple(latents_mean)
        latents_std = tuple(latents_std)
        temperal_upsample = tuple(reversed(temperal_downsample))

        h_img, w_img = (
            sample_size
            if isinstance(sample_size, (tuple, list))
            else (sample_size, sample_size)
        )
        spatial_compression_ratio = 2 ** len(temperal_downsample)
        h_lat, w_lat = (
            h_img // spatial_compression_ratio,
            w_img // spatial_compression_ratio,
        )

        encoder = QwenImageEncoder3d(
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            input_channels=input_channels,
            module_path="encoder",
            apply_temporal=apply_temporal,
        )
        decoder = QwenImageDecoder3d(
            dim=base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=temperal_upsample,
            dropout=dropout,
            input_channels=input_channels,
            module_path="decoder",
            apply_temporal=apply_temporal,
        )
        quant_conv = QwenImageCausalConv3d(
            z_dim * 2, kernel_size=1, padding=0, module_path="quant_conv"
        )
        post_quant_conv = QwenImageCausalConv3d(
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
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_upsample
        self.dropout = dropout
        self.input_channels = input_channels
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        self.sample_size = sample_size
        self.apply_temporal = apply_temporal
        self.spatial_compression_ratio = spatial_compression_ratio
        self.vae_scale_factor = spatial_compression_ratio
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv

    def _to_ndhwc(self, x, is_latent=False):
        """Normalize inputs to NDHWC ``(B, T, H, W, C)``."""
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

    def _maybe_squeeze_t(self, x, original_was_4d):
        if original_was_4d:
            return ops.squeeze(x, axis=1)
        return x

    def encode(self, x, sample=False, seed=None, return_ncdhw=False):
        """Encode image(s) to latents (mean, or reparameterized sample).

        Args:
            x: ``(B, H, W, 3)`` HWC, ``(B, 1, H, W, 3)`` NDHWC, or
                ``(B, 3, 1, H, W)`` Diffusers NCDHW.
            sample: If True, draw ``z ~ N(mean, std)``; else return mean.
            seed: RNG seed for sampling.
            return_ncdhw: If True, return Diffusers layout ``(B, C, T, H, W)``.
        """
        was_4d = len(x.shape) == 4
        x5 = self._to_ndhwc(x, is_latent=False)
        moments = self.quant_conv(self.encoder(x5))
        mean, logvar = ops.split(moments, 2, axis=-1)
        if sample:
            logvar = ops.clip(logvar, -30.0, 20.0)
            std = ops.exp(0.5 * logvar)
            noise = keras.random.normal(ops.shape(mean), dtype=mean.dtype, seed=seed)
            z = mean + std * noise
        else:
            z = mean
        if return_ncdhw:
            return ops.transpose(z, (0, 4, 1, 2, 3))
        return self._maybe_squeeze_t(z, was_4d)

    def decode(self, z, return_ncdhw=False):
        """Decode latents to RGB in ``[-1, 1]``.

        Args:
            z: ``(B, h, w, z_dim)``, ``(B, 1, h, w, z_dim)`` NDHWC, or
                ``(B, z_dim, 1, h, w)`` NCDHW.
            return_ncdhw: If True, return Diffusers layout.
        """
        was_4d = len(z.shape) == 4
        z5 = self._to_ndhwc(z, is_latent=True)
        x = self.decoder(self.post_quant_conv(z5))
        x = ops.clip(x, -1.0, 1.0)
        if return_ncdhw:
            return ops.transpose(x, (0, 4, 1, 2, 3))
        return self._maybe_squeeze_t(x, was_4d)

    def get_config(self):
        config = super().get_config()
        config.update(self.config.constructor_kwargs())
        config["apply_temporal"] = self.apply_temporal
        return config

    @classmethod
    def from_diffusers_config(cls, config, sample_size=1024, **kwargs):
        return cls(
            base_dim=config.get("base_dim", 96),
            z_dim=config.get("z_dim", 16),
            dim_mult=tuple(config.get("dim_mult", (1, 2, 4, 4))),
            num_res_blocks=config.get("num_res_blocks", 2),
            attn_scales=tuple(config.get("attn_scales", ())),
            temperal_downsample=tuple(
                config.get("temperal_downsample", (False, True, True))
            ),
            dropout=config.get("dropout", 0.0),
            input_channels=config.get("input_channels", 3),
            latents_mean=tuple(config.get("latents_mean", DEFAULT_LATENTS_MEAN)),
            latents_std=tuple(config.get("latents_std", DEFAULT_LATENTS_STD)),
            sample_size=sample_size,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageTransformer2DModel(BaseModel):
    """Qwen-Image denoiser (Diffusers ``QwenImageTransformer2DModel``).

    Double-stream MMDiT over **packed** latents ``(B, seq, in_channels)`` and
    text features ``(B, text_seq, joint_attention_dim)``. Inputs dict keys:
    ``sample``, ``timestep``, ``encoder_hidden_states``, optional
    ``encoder_hidden_states_mask``. Output ``{"sample": packed velocity}``.

    Built for a fixed packed sequence length derived from ``sample_size``
    (latent side before 2×2 packing).
    """

    HF_MODEL_TYPE = None
    config_class = QwenImageTransformerConfig

    def __init__(
        self,
        patch_size=2,
        in_channels=64,
        out_channels=16,
        num_layers=60,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=3584,
        axes_dims_rope=(16, 56, 56),
        guidance_embeds=False,
        sample_size=128,
        text_seq_len=512,
        name="QwenImageTransformer2DModel",
        **kwargs,
    ):
        del guidance_embeds  # base Qwen-Image is not guidance-distilled
        keras_kwargs = {k: kwargs.pop(k) for k in ("trainable", "dtype") if k in kwargs}
        axes_dims_rope = tuple(axes_dims_rope)
        inner_dim = num_attention_heads * attention_head_dim
        sample_h = (
            sample_size[0] if isinstance(sample_size, (tuple, list)) else sample_size
        )
        pack_h = pack_w = sample_h // patch_size
        packed_seq = pack_h * pack_w

        pos_embed = QwenImageEmbedRope(
            theta=10000,
            axes_dim=axes_dims_rope,
            scale_rope=True,
            module_path="pos_embed",
        )
        time_text_embed = QwenImageTimestepProjEmbeddings(
            embedding_dim=inner_dim, module_path="time_text_embed"
        )
        txt_norm = QwenImageRMSNorm(eps=1e-6, module_path="txt_norm")
        img_in = layers.Dense(inner_dim, name=safe_name("img_in"))
        txt_in = layers.Dense(inner_dim, name=safe_name("txt_in"))
        blocks = [
            QwenImageTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                module_path=f"transformer_blocks.{i}",
            )
            for i in range(num_layers)
        ]
        norm_out = QwenImageAdaLayerNormContinuous(
            inner_dim, num_chunks=2, module_path="norm_out"
        )
        proj_out = layers.Dense(
            patch_size * patch_size * out_channels,
            name=safe_name("proj_out"),
        )

        sample_in = layers.Input(shape=(packed_seq, in_channels), name="sample")
        timestep_in = layers.Input(shape=(), name="timestep")
        enc_in = layers.Input(
            shape=(text_seq_len, joint_attention_dim), name="encoder_hidden_states"
        )
        enc_mask_in = layers.Input(
            shape=(text_seq_len,), dtype="int32", name="encoder_hidden_states_mask"
        )

        hidden = img_in(sample_in)
        encoder = txt_in(txt_norm(enc_in))
        temb = time_text_embed(timestep_in)
        pos_embed.build(None)
        vid_freqs, txt_freqs = pos_embed.call(pack_h, pack_w, text_seq_len, frame=1)
        # Bake RoPE tables into the graph as constants (static T2I resolution).
        image_rotary_emb = (
            ops.convert_to_tensor(ops.convert_to_numpy(vid_freqs)),
            ops.convert_to_tensor(ops.convert_to_numpy(txt_freqs)),
        )
        mask = ops.cast(enc_mask_in, "bool")
        for block in blocks:
            encoder, hidden = block(
                hidden,
                encoder,
                temb,
                encoder_hidden_states_mask=mask,
                image_rotary_emb=image_rotary_emb,
            )
        hidden = norm_out([hidden, temb])
        output = proj_out(hidden)

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
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.axes_dims_rope = axes_dims_rope
        self.sample_size = sample_size
        self.text_seq_len = text_seq_len
        self.inner_dim = inner_dim
        self.pack_h = pack_h
        self.pack_w = pack_w
        self.packed_seq = packed_seq
        self.pos_embed = pos_embed
        self.time_text_embed = time_text_embed
        self.txt_norm = txt_norm
        self.img_in = img_in
        self.txt_in = txt_in
        self.transformer_blocks = blocks
        self.norm_out = norm_out
        self.proj_out = proj_out

    @classmethod
    def kwargs_from_diffusers_config(cls, cfg):
        return {
            "patch_size": cfg.get("patch_size", 2),
            "in_channels": cfg.get("in_channels", 64),
            "out_channels": cfg.get("out_channels", 16),
            "num_layers": cfg.get("num_layers", 60),
            "attention_head_dim": cfg.get("attention_head_dim", 128),
            "num_attention_heads": cfg.get("num_attention_heads", 24),
            "joint_attention_dim": cfg.get("joint_attention_dim", 3584),
            "axes_dims_rope": tuple(cfg.get("axes_dims_rope", (16, 56, 56))),
            "guidance_embeds": bool(cfg.get("guidance_embeds", False)),
        }


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageTextEncoderModel(Qwen2_5VLModel):
    """Qwen-Image prompt encoder: the Qwen2.5-VL-7B text tower, no vision / LM head.

    The ``text_encoder`` component of a Qwen-Image :class:`ZMDiffusion`.
    Inputs ``input_ids`` / ``attention_mask``; output ``last_hidden_state``.
    """

    HF_MODEL_TYPE = None
    config_class = QwenImageTextConfig

    def __init__(self, max_seq_len=1024, name="text_encoder", **kwargs):
        kwargs["build_vision"] = False
        super().__init__(name=name, **kwargs)
        self.max_seq_len = max_seq_len


class QwenImageFamily:
    """Qwen-Image inside :class:`ZMDiffusion` (``model_type="qwen_image"``).

    Three components, ``transformer`` (the double-stream MMDiT,
    :class:`QwenImageTransformer2DModel`), ``vae`` (the Wan-derived
    :class:`AutoencoderKLQwenImage`) and ``text_encoder`` (the Qwen2.5-VL-7B text
    tower), wired into three disconnected paths (diffusers' ``QwenImagePipeline``
    components). Hosted as ``zeromodels/qwen-image`` (sharded weights).

    Generation follows the reference: the ChatML-templated prompt through the text
    tower with the template prefix dropped, packed 2x2 latents, the flow-match Euler
    sampler with resolution-dependent shifting, and true classifier-free guidance
    (``guidance_scale``, diffusers' ``true_cfg_scale``: separate conditional and
    unconditional passes, the combination renormalized to the conditional
    prediction's norm). Without a negative prompt the unconditional branch is the
    templated blank prompt. ``height`` / ``width`` choose the output size.
    """

    model_type = "qwen_image"
    config_class = QwenImageConfig
    generate_args = {"num_inference_steps": 50, "guidance_scale": 4.0}

    def build_components(self, config, data_format, channels_axis):
        d, v = config.transformer_config, config.vae_config
        transformer = QwenImageTransformer2DModel(
            patch_size=d.patch_size,
            in_channels=d.in_channels,
            out_channels=d.out_channels,
            num_layers=d.num_layers,
            attention_head_dim=d.attention_head_dim,
            num_attention_heads=d.num_attention_heads,
            joint_attention_dim=d.joint_attention_dim,
            axes_dims_rope=d.axes_dims_rope,
            guidance_embeds=d.guidance_embeds,
            sample_size=d.sample_size,
            text_seq_len=config.max_sequence_length,
        )
        vae = AutoencoderKLQwenImage(
            base_dim=v.base_dim,
            z_dim=v.z_dim,
            dim_mult=v.dim_mult,
            num_res_blocks=v.num_res_blocks,
            attn_scales=v.attn_scales,
            temperal_downsample=v.temperal_downsample,
            dropout=v.dropout,
            input_channels=v.input_channels,
            latents_mean=v.latents_mean,
            latents_std=v.latents_std,
            sample_size=v.sample_size,
        )
        text_encoder = QwenImageTextEncoderModel(config.text_config)
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
                shape=(transformer.packed_seq, d.in_channels), name="sample"
            ),
            "timestep": layers.Input(shape=(), name="timestep"),
            "encoder_hidden_states": layers.Input(
                shape=(text_seq, d.joint_attention_dim), name="encoder_hidden_states"
            ),
            "encoder_hidden_states_mask": layers.Input(
                shape=(text_seq,), dtype="int32", name="encoder_hidden_states_mask"
            ),
            "image": layers.Input(shape=(img_h, img_w, 3), name="image"),
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

    def default_scheduler(self):
        return FlowMatchEulerDiscreteScheduler(
            shift=1.0,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=0.9,
            base_image_seq_len=256,
            max_image_seq_len=8192,
            shift_terminal=0.02,
            time_shift_type="exponential",
        )

    def latent_side(self, model, height=None, width=None):
        # diffusers: round the size down to a multiple of vae_scale_factor * 2 (the
        # VAE downsampling times the 2x2 packing), in latent pixels
        scale = model.vae.vae_scale_factor * 2
        if height is None or width is None:
            side = model.config.default_sample_size * model.vae.vae_scale_factor
            height = width = side
        return 2 * (int(height) // scale), 2 * (int(width) // scale)

    def generation_side(self, model):
        height, width = model.__dict__.get("generation_size", (None, None))
        return self.latent_side(model, height, width)

    def image_seq_len(self, height, width):
        return (height // 2) * (width // 2)

    def latent_shape(self, model):
        # the packed shape, at the default resolution
        h, w = self.latent_side(model)
        return (self.image_seq_len(h, w), model.vae.z_dim * 4)

    def pack_latents(self, latents, height, width):
        return pack_latents(latents, height, width)

    def unpack_latents(self, latents, height, width, channels):
        return unpack_latents(latents, height, width, channels)

    def negative_prompt_inputs(self, model, batch, prompt=None):
        # the templated negative prompt of true CFG (a blank prompt by default),
        # tokenized with the hosted Qwen-Image tokenizer
        from zeromodels.diffusers.diffusion_tokenizer import ZMDiffusionTokenizer

        if prompt is None:
            prompts = [" "] * batch
        elif isinstance(prompt, str):
            prompts = [prompt] * batch
        else:
            prompts = list(prompt)
        if len(prompts) != batch:
            raise ValueError("negative_prompt must have one entry per image")
        tokenizer = model.__dict__.get("negative_prompt_tokenizer")
        if tokenizer is None:
            tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/qwen-image")
            model.__dict__["negative_prompt_tokenizer"] = tokenizer
        return tokenizer(prompts)

    def unconditional_ids(self, model, batch):
        return ops.convert_to_tensor(
            self.negative_prompt_inputs(model, batch)["input_ids"], dtype="int32"
        )

    def encode_prompt(self, model, input_ids, attention_mask=None, **conditioning):
        # expects ids already wrapped in the ChatML prompt template (as
        # ZMDiffusionTokenizer returns them); drops the template prefix
        # (prompt_template_encode_start_idx) and pads / truncates to
        # max_sequence_length
        del conditioning
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        else:
            attention_mask = ops.cast(ops.convert_to_tensor(attention_mask), "int32")

        out = model.text_encoder(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        hidden = out["last_hidden_state"]
        drop = int(model.config.prompt_template_encode_start_idx)
        max_len = int(model.config.max_sequence_length)

        # gather each row's real tokens on the host, drop the template prefix, pad
        hidden_np = ops.convert_to_numpy(hidden)
        mask_np = ops.convert_to_numpy(attention_mask)
        batch = hidden_np.shape[0]
        dim = hidden_np.shape[-1]
        embeds = np.zeros((batch, max_len, dim), dtype=hidden_np.dtype)
        out_mask = np.zeros((batch, max_len), dtype=np.int32)
        for i in range(batch):
            valid = hidden_np[i][mask_np[i].astype(bool)]
            valid = valid[drop:][:max_len]
            n = valid.shape[0]
            embeds[i, :n] = valid
            out_mask[i, :n] = 1
        return {
            "encoder_hidden_states": ops.convert_to_tensor(embeds),
            "encoder_hidden_states_mask": ops.convert_to_tensor(out_mask),
        }

    def predict_noise(self, model, latents, timesteps, embeddings):
        inputs = {
            "sample": latents,
            "timestep": timesteps,
            "encoder_hidden_states": embeddings["encoder_hidden_states"],
            "encoder_hidden_states_mask": embeddings["encoder_hidden_states_mask"],
        }
        impl = getattr(model, "_attn_implementation", None)
        if impl is None:
            return model.transformer(inputs)["sample"]
        with use_attn_implementation(impl):
            return model.transformer(inputs)["sample"]

    def decode_latents(self, model, latents, height=None, width=None):
        h, w = self.latent_side(model, height, width)
        latents = self.unpack_latents(latents, h, w, model.vae.z_dim)
        # diffusers: latents / (1 / std) + mean == latents * std + mean
        mean = ops.convert_to_tensor(model.vae.latents_mean, dtype="float32")
        std = ops.convert_to_tensor(model.vae.latents_std, dtype="float32")
        mean = ops.reshape(mean, (1, 1, 1, -1))
        std = ops.reshape(std, (1, 1, 1, -1))
        latents = latents * std + mean
        return model.vae.decode(latents)

    def prepare_latents(self, model, batch, seed=None, latents=None, dtype="float32"):
        h, w = self.generation_side(model)
        if latents is None:
            # spatial noise, then packed (diffusers' prepare_latents)
            noise = keras.random.normal(
                (batch, h, w, model.vae.z_dim), seed=seed, dtype=dtype
            )
            latents = self.pack_latents(noise, h, w)
        else:
            latents = ops.cast(ops.convert_to_tensor(latents), dtype)
        return latents * model.scheduler.init_noise_sigma

    def combine_guidance(self, noise_pred, neg_pred, guidance_scale):
        comb = neg_pred + guidance_scale * (noise_pred - neg_pred)
        cond_norm = ops.sqrt(ops.sum(ops.square(noise_pred), axis=-1, keepdims=True))
        comb_norm = ops.sqrt(ops.sum(ops.square(comb), axis=-1, keepdims=True))
        return comb * (cond_norm / (comb_norm + 1e-8))

    def denoise(
        self,
        model,
        latents,
        embeddings,
        num_inference_steps,
        guidance_scale,
        timesteps=None,
    ):
        # true CFG: separate conditional / unconditional passes; embeddings is the
        # (uncond, cond) pair under guidance
        do_cfg = guidance_scale > 1.0
        scheduler = model.scheduler
        if timesteps is None:
            scheduler.set_timesteps(num_inference_steps)
            timesteps = scheduler.timesteps
        if do_cfg and isinstance(embeddings, (tuple, list)):
            uncond_emb, cond_emb = embeddings
        else:
            cond_emb = embeddings
            uncond_emb = None

        for t in timesteps:
            t_batch = ops.full((ops.shape(latents)[0],), float(t) / 1000.0)
            noise_pred = model.predict_noise(latents, t_batch, cond_emb)
            if do_cfg and uncond_emb is not None:
                neg_pred = model.predict_noise(latents, t_batch, uncond_emb)
                noise_pred = self.combine_guidance(noise_pred, neg_pred, guidance_scale)
            latents = scheduler.step(noise_pred, t, latents)
        return latents

    def negative_embeddings(
        self,
        model,
        batch,
        negative_input_ids,
        negative_attention_mask,
        guidance_scale,
        conditioning,
    ):
        # the unconditional branch, or None to run without guidance
        if guidance_scale <= 1.0:
            return None
        if negative_input_ids is None:
            negative_inputs = self.negative_prompt_inputs(
                model, batch, conditioning.get("negative_prompt")
            )
            negative_input_ids = negative_inputs["input_ids"]
            negative_attention_mask = negative_inputs["attention_mask"]
        return model.encode_prompt(negative_input_ids, negative_attention_mask)

    def set_timesteps(self, model, num_inference_steps, height, width):
        # diffusers' calculate_shift: the resolution-dependent flow-match mu
        h, w = self.latent_side(model, height, width)
        image_seq_len = self.image_seq_len(h, w)
        sched_cfg = getattr(model.scheduler, "config_dict", None) or {}
        base_seq_len = sched_cfg.get("base_image_seq_len", 256)
        max_seq_len = sched_cfg.get("max_image_seq_len", 4096)
        base_shift = sched_cfg.get("base_shift", 0.5)
        max_shift = sched_cfg.get("max_shift", 0.9)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        mu = image_seq_len * m + (base_shift - m * base_seq_len)
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        if hasattr(model.scheduler, "set_timesteps"):
            try:
                model.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, mu=mu)
            except TypeError:
                model.scheduler.set_timesteps(num_inference_steps)
        return model.scheduler.timesteps

    def generate(
        self,
        model,
        input_ids,
        attention_mask=None,
        negative_input_ids=None,
        negative_attention_mask=None,
        num_inference_steps=None,
        guidance_scale=None,
        seed=None,
        latents=None,
        height=None,
        width=None,
        output_type="image",
        **conditioning,
    ):
        num_inference_steps, guidance_scale, seed = model.resolve_generation_args(
            num_inference_steps, guidance_scale, seed
        )
        model.__dict__["generation_size"] = (height, width)
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch = int(input_ids.shape[0])

        with inference_scope():
            embeddings = model.encode_prompt(input_ids, attention_mask, **conditioning)
            uncond = self.negative_embeddings(
                model,
                batch,
                negative_input_ids,
                negative_attention_mask,
                guidance_scale,
                conditioning,
            )
            do_cfg = uncond is not None
            timesteps = self.set_timesteps(model, num_inference_steps, height, width)
            latents = model.prepare_latents(batch, seed=seed, latents=latents)
            latents = model.denoise(
                latents,
                (uncond, embeddings) if do_cfg else embeddings,
                num_inference_steps,
                guidance_scale if do_cfg else 1.0,
                timesteps=timesteps,
            )
            if output_type == "latent":
                return latents
            image = model.decode_latents(latents, height=height, width=width)
        return model.postprocess_image(image)
