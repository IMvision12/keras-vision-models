import keras
from keras import layers, ops

from zeromodels.base import BaseModel
from zeromodels.base.base_diffusion import BaseDiffusion
from zeromodels.base.base_mixin import inference_scope
from zeromodels.base.base_scheduler import FlowMatchEulerDiscreteScheduler
from zeromodels.diffusers.stable_diffusion.stable_diffusion_layers import (
    time_embedding_mlp,
    timestep_embedding,
)
from zeromodels.diffusers.stable_diffusion.stable_diffusion_model import (
    AutoencoderKL,
    StableDiffusionFamily,
)
from zeromodels.diffusers.stable_diffusion_xl.stable_diffusion_xl_model import (
    build_text_encoder,
)
from zeromodels.models.t5.t5_layers import T5LayerNorm
from zeromodels.models.t5.t5_model import T5PositionBias

from .stable_diffusion_3_config import (
    StableDiffusion3Config,
    StableDiffusion3T5EncoderConfig,
    StableDiffusion3TransformerConfig,
)
from .stable_diffusion_3_layers import (
    StableDiffusion3AdaLayerNorm,
    StableDiffusion3JointTransformerBlock,
    StableDiffusion3PatchEmbed,
    StableDiffusion3T5GatedEncoderBlock,
)


@keras.saving.register_keras_serializable(package="zeromodels")
class SD3Transformer2DModel(BaseModel):
    """Stable Diffusion 3 denoiser, the MMDiT (diffusers ``SD3Transformer2DModel``).

    A multimodal diffusion transformer: the latent is patchified into tokens with
    a cropped sinusoidal position table, the text tokens are projected to the same
    width, and ``num_layers`` joint blocks let the two streams attend to each other
    while a conditioning embedding (the sinusoidal timestep plus the pooled text
    embedding) modulates every norm (AdaLN-Zero). The final norm and projection
    unpatchify the latent tokens back to a velocity field of the latent's shape.

    Inputs are a dict ``{"sample", "timestep", "encoder_hidden_states",
    "pooled_projections"}``; the output dict is ``{"sample": (B, H, W, C)}``.
    Built for a fixed latent resolution (``sample_size``) like the SD UNet; the
    weights are resolution-independent up to ``pos_embed_max_size`` patches.

    Args mirror the diffusers config: ``sample_size`` (128), ``patch_size`` (2),
    ``in_channels`` / ``out_channels`` (16), ``num_layers`` (24), ``attention_head_dim``
    (64), ``num_attention_heads`` (24), ``joint_attention_dim`` (4096, the text
    features' width), ``caption_projection_dim`` (the projected text width, the
    token width), ``pooled_projection_dim`` (2048), ``pos_embed_max_size`` (192),
    ``qk_norm`` (``"rms_norm"`` in SD 3.5), ``dual_attention_layers`` (the SD 3.5
    medium blocks with a second latent-only attention) and ``text_seq_len`` (the
    static text length, 77 + 256).
    """

    HF_MODEL_TYPE = None
    config_class = StableDiffusion3TransformerConfig

    def __init__(
        self,
        sample_size=128,
        patch_size=2,
        in_channels=16,
        out_channels=16,
        num_layers=24,
        attention_head_dim=64,
        num_attention_heads=24,
        joint_attention_dim=4096,
        caption_projection_dim=1536,
        pooled_projection_dim=2048,
        pos_embed_max_size=192,
        qk_norm=None,
        dual_attention_layers=(),
        text_seq_len=333,
        data_format=None,
        channels_axis=None,
        name="SD3Transformer2DModel",
        **kwargs,
    ):
        data_format = data_format or keras.config.image_data_format()
        if channels_axis is None:
            channels_axis = -1 if data_format == "channels_last" else 1
        inner_dim = num_attention_heads * attention_head_dim
        sample_h, sample_w = (
            sample_size
            if isinstance(sample_size, (tuple, list))
            else (sample_size, sample_size)
        )
        dual_attention_layers = tuple(dual_attention_layers)

        sample_in = layers.Input(
            shape=(in_channels, sample_h, sample_w)
            if data_format == "channels_first"
            else (sample_h, sample_w, in_channels),
            name="sample",
        )
        timestep_in = layers.Input(shape=(), name="timestep")
        context_in = layers.Input(
            shape=(text_seq_len, joint_attention_dim), name="encoder_hidden_states"
        )
        pooled_in = layers.Input(
            shape=(pooled_projection_dim,), name="pooled_projections"
        )

        x = StableDiffusion3PatchEmbed(
            inner_dim,
            patch_size,
            pos_embed_max_size,
            base_size=sample_h // patch_size,
            module_path="pos_embed",
            data_format=data_format,
            channels_axis=channels_axis,
        )(sample_in)
        temb = timestep_embedding(timestep_in, 256)
        temb = time_embedding_mlp(
            temb, inner_dim, name="time_text_embed.timestep_embedder"
        )
        temb = temb + time_embedding_mlp(
            pooled_in, inner_dim, name="time_text_embed.text_embedder"
        )
        context = layers.Dense(caption_projection_dim, name="context_embedder")(
            context_in
        )
        for i in range(num_layers):
            block = StableDiffusion3JointTransformerBlock(
                inner_dim,
                num_attention_heads,
                module_path=f"transformer_blocks.{i}",
                context_pre_only=i == num_layers - 1,
                qk_norm=qk_norm,
                use_dual_attention=i in dual_attention_layers,
            )
            if i == num_layers - 1:
                x = block([x, context, temb])
            else:
                x, context = block([x, context, temb])
        x = StableDiffusion3AdaLayerNorm(inner_dim, 2, module_path="norm_out")(
            [x, temb]
        )
        x = layers.Dense(patch_size * patch_size * out_channels, name="proj_out")(x)

        # unpatchify: (B, h*w, p*p*C) -> (B, h*p, w*p, C)
        h, w = sample_h // patch_size, sample_w // patch_size
        x = ops.reshape(x, (-1, h, w, patch_size, patch_size, out_channels))
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (-1, h * patch_size, w * patch_size, out_channels))
        if data_format == "channels_first":
            x = ops.transpose(x, (0, 3, 1, 2))

        super().__init__(
            inputs={
                "sample": sample_in,
                "timestep": timestep_in,
                "encoder_hidden_states": context_in,
                "pooled_projections": pooled_in,
            },
            outputs={"sample": x},
            name=name,
            **kwargs,
        )

        self.data_format = data_format
        self.channels_axis = channels_axis
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.caption_projection_dim = caption_projection_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.pos_embed_max_size = pos_embed_max_size
        self.qk_norm = qk_norm
        self.dual_attention_layers = dual_attention_layers
        self.text_seq_len = text_seq_len

    @classmethod
    def from_diffusers_config(cls, config, **kwargs):
        """Build from a diffusers ``transformer/config.json`` dict."""
        return cls(**cls.kwargs_from_diffusers_config(config), **kwargs)

    @staticmethod
    def kwargs_from_diffusers_config(config):
        """The constructor kwargs a diffusers ``transformer/config.json`` describes."""
        return {
            "sample_size": config.get("sample_size", 128),
            "patch_size": config.get("patch_size", 2),
            "in_channels": config.get("in_channels", 16),
            "out_channels": config.get("out_channels") or config.get("in_channels", 16),
            "num_layers": config.get("num_layers", 24),
            "attention_head_dim": config.get("attention_head_dim", 64),
            "num_attention_heads": config.get("num_attention_heads", 24),
            "joint_attention_dim": config.get("joint_attention_dim", 4096),
            "caption_projection_dim": config.get("caption_projection_dim", 1536),
            "pooled_projection_dim": config.get("pooled_projection_dim", 2048),
            "pos_embed_max_size": config.get("pos_embed_max_size", 192),
            "qk_norm": config.get("qk_norm"),
            "dual_attention_layers": tuple(config.get("dual_attention_layers") or ()),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sample_size": self.sample_size,
                "patch_size": self.patch_size,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "num_layers": self.num_layers,
                "attention_head_dim": self.attention_head_dim,
                "num_attention_heads": self.num_attention_heads,
                "joint_attention_dim": self.joint_attention_dim,
                "caption_projection_dim": self.caption_projection_dim,
                "pooled_projection_dim": self.pooled_projection_dim,
                "pos_embed_max_size": self.pos_embed_max_size,
                "qk_norm": self.qk_norm,
                "dual_attention_layers": self.dual_attention_layers,
                "text_seq_len": self.text_seq_len,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SD3T5EncoderModel(BaseModel):
    """The third Stable Diffusion 3 text encoder: the encoder half of T5 v1.1 XXL
    (transformers ``T5EncoderModel`` on ``google/t5-v1_1-xxl``), pure Keras 3.

    A shared token embedding, ``num_layers`` pre-norm blocks (bias-free self-attention
    without ``1/sqrt(d)`` scaling, one learned relative position bias shared by all
    blocks, the gated-GELU feed-forward of T5 v1.1) and a final RMSNorm. The
    attention, norm and position-bias layers are the t5 family's; the gated
    feed-forward, which the original T5 does not have, is
    :class:`~zeromodels.diffusers.stable_diffusion_3.stable_diffusion_3_layers.StableDiffusion3T5GatedFeedForward`.
    Inputs ``{"input_ids", "attention_mask"}`` (``(B, T)`` int32), output
    ``{"last_hidden_state": (B, T, embed_dim)}``.

    Every SD 3 / 3.5 checkpoint conditions on the same 4.7B-parameter encoder, so it
    is hosted once (``zeromodels/t5-v1_1-xxl-encoder``, float16 weights) and attached
    to a :class:`ZMDiffusion` on demand:
    ``ZMDiffusion.from_weights(repo, text_encoder_3="zeromodels/t5-v1_1-xxl-encoder")``.

    Args:
        vocab_size: SentencePiece vocabulary (32128).
        embed_dim: Token width (4096).
        key_value_dim: Width of one attention head (64).
        mlp_dim: Feed-forward inner width (10240).
        num_layers: Encoder blocks (24).
        num_heads: Attention heads (64).
        relative_attention_num_buckets: Relative-position-bias buckets (32).
        relative_attention_max_distance: Largest bucketed distance (128).
        layer_norm_eps: RMSNorm epsilon.
        pad_token_id / eos_token_id: The tokenizer's ``<pad>`` and ``</s>`` ids.
    """

    HF_MODEL_TYPE = None
    config_class = StableDiffusion3T5EncoderConfig

    def __init__(
        self,
        vocab_size=32128,
        embed_dim=4096,
        key_value_dim=64,
        mlp_dim=10240,
        num_layers=24,
        num_heads=64,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        layer_norm_eps=1e-6,
        pad_token_id=0,
        eos_token_id=1,
        name="SD3T5EncoderModel",
        **kwargs,
    ):
        shared = layers.Embedding(vocab_size, embed_dim, name="shared")
        rel_bias = layers.Embedding(
            relative_attention_num_buckets, num_heads, name="encoder_rel_bias"
        )
        bias = T5PositionBias(
            rel_bias,
            num_heads,
            relative_attention_num_buckets,
            relative_attention_max_distance,
            bidirectional=True,
            causal=False,
            name="encoder_bias",
        )
        blocks = [
            StableDiffusion3T5GatedEncoderBlock(
                embed_dim,
                key_value_dim,
                num_heads,
                mlp_dim,
                layer_norm_eps,
                prefix=f"enc_{i}",
                name=f"encoder_block_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = T5LayerNorm(layer_norm_eps, name="encoder_final_layer_norm")

        input_ids_in = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        attn_in = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
        hidden = shared(input_ids_in)
        position_bias = bias(hidden, attn_in)
        for block in blocks:
            hidden = block(hidden, position_bias)
        hidden = final_norm(hidden)
        super().__init__(
            inputs={"input_ids": input_ids_in, "attention_mask": attn_in},
            outputs={"last_hidden_state": hidden},
            name=name,
            **kwargs,
        )
        self.shared = shared
        self.encoder_rel_bias = rel_bias
        self.encoder_bias = bias
        self.encoder_blocks = blocks
        self.encoder_final_layer_norm = final_norm
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.key_value_dim = key_value_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        # the relative bias table is read inside the position-bias layer at run
        # time (not traced into the graph): one eager call creates its weight so
        # the checkpoint can load into it
        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )

    @classmethod
    def from_hf_config(cls, hf_config, **kwargs):
        """Build from a transformers ``T5Config`` dict (``text_encoder_3/config.json``)."""
        return cls(**cls.kwargs_from_hf_config(hf_config), **kwargs)

    @staticmethod
    def kwargs_from_hf_config(hf_config):
        """The constructor kwargs a transformers T5 v1.1 ``config.json`` describes."""
        return {
            "vocab_size": hf_config.get("vocab_size", 32128),
            "embed_dim": hf_config.get("d_model", 4096),
            "key_value_dim": hf_config.get("d_kv", 64),
            "mlp_dim": hf_config.get("d_ff", 10240),
            "num_layers": hf_config.get("num_layers", 24),
            "num_heads": hf_config.get("num_heads", 64),
            "relative_attention_num_buckets": hf_config.get(
                "relative_attention_num_buckets", 32
            ),
            "relative_attention_max_distance": hf_config.get(
                "relative_attention_max_distance", 128
            ),
            "layer_norm_eps": hf_config.get("layer_norm_epsilon", 1e-6),
            "pad_token_id": hf_config.get("pad_token_id", 0),
            "eos_token_id": hf_config.get("eos_token_id", 1),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "key_value_dim": self.key_value_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "relative_attention_num_buckets": self.relative_attention_num_buckets,
                "relative_attention_max_distance": self.relative_attention_max_distance,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "eos_token_id": self.eos_token_id,
                "name": self.name,
            }
        )
        return config


class StableDiffusion3Family(StableDiffusionFamily):
    """Stable Diffusion 3 inside :class:`ZMDiffusion` (``model_type="stable_diffusion_3"``).

    Four components, ``transformer`` (the MMDiT, :class:`SD3Transformer2DModel`),
    ``vae`` (16 channels) and the two CLIP towers ``text_encoder`` (ViT-L/14) /
    ``text_encoder_2`` (OpenCLIP ViT-bigG/14), each with its projection, wired into
    four disconnected paths: ``{"sample", "timestep", "encoder_hidden_states",
    "pooled_projections"} -> "noise_pred"``, ``{"image", "latent"} -> "moments" /
    "image"`` and ``{"token_ids", "token_ids_2", "padding_mask"} -> "prompt_embeds" /
    "pooled_prompt_embeds"`` (penultimate states concatenated to 2048-d, projected
    pooled states concatenated to 2048-d).

    The third text encoder, the 4.7B-parameter T5-XXL, is not part of the checkpoint:
    every SD 3 / 3.5 checkpoint shares it, so it is hosted once
    (``zeromodels/t5-v1_1-xxl-encoder``, an :class:`SD3T5EncoderModel`) and attached
    with ``ZMDiffusion.from_weights(repo, text_encoder_3=...)`` or by assigning
    ``model.text_encoder_3``. Without it the T5 features are zeros, SD 3's documented
    memory-saving mode.

    Generation follows diffusers' ``StableDiffusion3Pipeline`` with the rectified-flow
    sampler (:class:`FlowMatchEulerDiscreteScheduler`): the CLIP states, zero-padded
    to the T5 width, followed by the T5 tokens (``input_ids_3``, 256) along the
    sequence, the pooled CLIP states along the features. With no negative prompt the
    empty prompt is encoded (no zeroing).
    """

    model_type = "stable_diffusion_3"
    config_class = StableDiffusion3Config
    generate_args = {"num_inference_steps": 28, "guidance_scale": 7.0}
    supports_text_encoder_3 = True

    def build_components(self, config, data_format, channels_axis):
        d, v, t, t2 = (
            config.transformer_config,
            config.vae_config,
            config.text_config,
            config.text_config_2,
        )
        transformer = SD3Transformer2DModel(
            d,
            text_seq_len=t.max_seq_len + config.max_sequence_length,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        vae = AutoencoderKL(
            **v.constructor_kwargs(),
            data_format=data_format,
            channels_axis=channels_axis,
        )
        text_encoder = build_text_encoder(
            t,
            config.hidden_act,
            config.layer_norm_eps,
            projection_dim=t.projection_dim,
            name="text_encoder",
        )
        text_encoder_2 = build_text_encoder(
            t2,
            t2.hidden_act,
            config.layer_norm_eps,
            projection_dim=t2.projection_dim,
            name="text_encoder_2",
        )
        return {
            "transformer": transformer,
            "vae": vae,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
        }

    def build_graph(self, config, components, data_format):
        d, v, t = config.transformer_config, config.vae_config, config.text_config
        transformer, vae = components["transformer"], components["vae"]
        text_encoder, text_encoder_2 = (
            components["text_encoder"],
            components["text_encoder_2"],
        )
        sample_h, sample_w = (
            d.sample_size
            if isinstance(d.sample_size, (tuple, list))
            else (d.sample_size, d.sample_size)
        )
        img_h, img_w = (
            v.sample_size
            if isinstance(v.sample_size, (tuple, list))
            else (v.sample_size, v.sample_size)
        )
        lat_h, lat_w = img_h // vae.vae_scale_factor, img_w // vae.vae_scale_factor
        inputs = {
            "sample": layers.Input(
                shape=(d.in_channels, sample_h, sample_w)
                if data_format == "channels_first"
                else (sample_h, sample_w, d.in_channels),
                name="sample",
            ),
            "timestep": layers.Input(shape=(), name="timestep"),
            "encoder_hidden_states": layers.Input(
                shape=(transformer.text_seq_len, d.joint_attention_dim),
                name="encoder_hidden_states",
            ),
            "pooled_projections": layers.Input(
                shape=(d.pooled_projection_dim,), name="pooled_projections"
            ),
            "image": layers.Input(
                shape=(v.in_channels, img_h, img_w)
                if data_format == "channels_first"
                else (img_h, img_w, v.in_channels),
                name="image",
            ),
            "latent": layers.Input(
                shape=(v.latent_channels, lat_h, lat_w)
                if data_format == "channels_first"
                else (lat_h, lat_w, v.latent_channels),
                name="latent",
            ),
            "token_ids": layers.Input(
                shape=(t.max_seq_len,), dtype="int32", name="token_ids"
            ),
            "token_ids_2": layers.Input(
                shape=(t.max_seq_len,), dtype="int32", name="token_ids_2"
            ),
            "padding_mask": layers.Input(
                shape=(t.max_seq_len,), dtype="int32", name="padding_mask"
            ),
        }
        noise_pred = transformer(
            {
                key: inputs[key]
                for key in (
                    "sample",
                    "timestep",
                    "encoder_hidden_states",
                    "pooled_projections",
                )
            }
        )["sample"]
        vae_out = vae({"image": inputs["image"], "latent": inputs["latent"]})
        text_out = text_encoder(
            {"token_ids": inputs["token_ids"], "padding_mask": inputs["padding_mask"]}
        )
        text_out_2 = text_encoder_2(
            {
                "token_ids": inputs["token_ids_2"],
                "padding_mask": inputs["padding_mask"],
            }
        )
        outputs = {
            "noise_pred": noise_pred,
            "moments": vae_out["moments"],
            "image": vae_out["sample"],
            "prompt_embeds": ops.concatenate(
                [
                    text_out["penultimate_hidden_state"],
                    text_out_2["penultimate_hidden_state"],
                ],
                axis=-1,
            ),
            "pooled_prompt_embeds": ops.concatenate(
                [text_out["text_embeds"], text_out_2["text_embeds"]], axis=-1
            ),
        }
        return inputs, outputs

    def default_scheduler(self):
        return FlowMatchEulerDiscreteScheduler(shift=3.0)

    def latent_shape(self, model):
        size = model.transformer.sample_size
        height, width = size if isinstance(size, (tuple, list)) else (size, size)
        if model.data_format == "channels_first":
            return (model.transformer.in_channels, height, width)
        return (height, width, model.transformer.in_channels)

    def unconditional_ids_3(self, model, batch):
        cfg = model.config
        row = [cfg.eos_token_id_3] + [cfg.pad_token_id_3] * (
            cfg.max_sequence_length - 1
        )
        return ops.convert_to_tensor([row] * batch, dtype="int32")

    def prompt_mask(self, model, input_ids):
        # the tokens up to and including the first <|endoftext|>; what follows is
        # padding (the first tokenizer pads with <|endoftext|> itself)
        is_eos = ops.cast(ops.equal(input_ids, model.config.eos_token_id), "int32")
        return ops.equal(ops.cumsum(is_eos, axis=1) - is_eos, 0)

    def encode_prompt(self, model, input_ids, attention_mask=None, input_ids_3=None):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch = int(input_ids.shape[0])
        if attention_mask is None:
            real = self.prompt_mask(model, input_ids)
        else:
            real = ops.greater(
                ops.cast(ops.convert_to_tensor(attention_mask), "int32"), 0
            )
        # the second tokenizer pads with "!" (id 0) where the first pads with
        # <|endoftext|>; the towers attend the padding, as the reference does
        input_ids_2 = ops.where(real, input_ids, model.config.pad_token_id_2)
        ones = ops.ones_like(input_ids)
        out = model.text_encoder({"token_ids": input_ids, "padding_mask": ones})
        out_2 = model.text_encoder_2({"token_ids": input_ids_2, "padding_mask": ones})
        clip_embeds = ops.concatenate(
            [out["penultimate_hidden_state"], out_2["penultimate_hidden_state"]],
            axis=-1,
        )
        pooled = ops.concatenate([out["text_embeds"], out_2["text_embeds"]], axis=-1)
        # the CLIP features are zero-padded to the T5 width, then the T5 tokens
        # (or zeros without a T5 encoder) follow along the sequence
        t5_dim = model.transformer.joint_attention_dim
        clip_embeds = ops.pad(
            clip_embeds, ((0, 0), (0, 0), (0, t5_dim - clip_embeds.shape[-1]))
        )
        length = model.config.max_sequence_length
        if model.text_encoder_3 is None:
            t5_embeds = ops.zeros((batch, length, t5_dim), dtype=clip_embeds.dtype)
        else:
            if input_ids_3 is None:
                input_ids_3 = self.unconditional_ids_3(model, batch)
            input_ids_3 = ops.cast(ops.convert_to_tensor(input_ids_3), "int32")
            if int(input_ids_3.shape[0]) == 1 and batch > 1:
                input_ids_3 = ops.repeat(input_ids_3, batch, axis=0)  # one for all
            t5_embeds = model.text_encoder_3(
                {"input_ids": input_ids_3, "attention_mask": ops.ones_like(input_ids_3)}
            )["last_hidden_state"]
            t5_embeds = ops.cast(t5_embeds, clip_embeds.dtype)
        return {
            "encoder_hidden_states": ops.concatenate([clip_embeds, t5_embeds], axis=1),
            "pooled_projections": pooled,
        }

    def encode_negative_prompt(self, model, negative_input_ids, batch, **conditioning):
        if negative_input_ids is None:
            # the empty prompt for the T5 tower as well, not the positive prompt's ids
            conditioning = dict(conditioning, input_ids_3=None)
        return BaseDiffusion.encode_negative_prompt(
            model, negative_input_ids, batch, **conditioning
        )

    def predict_noise(self, model, latents, timesteps, embeddings):
        return model.transformer(
            {"sample": latents, "timestep": timesteps, **embeddings}
        )["sample"]
