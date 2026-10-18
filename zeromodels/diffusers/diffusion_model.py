import functools

import keras

from zeromodels.base import BaseConfig, BaseModel
from zeromodels.base.base_diffusion import BaseDiffusion
from zeromodels.base.base_model import strip_functional_graph_keys
from zeromodels.base.base_scheduler import get_scheduler

from .qwen_image.qwen_image_model import QwenImageFamily
from .qwen_image_21.qwen_image_21_model import QwenImage21Family
from .stable_diffusion.stable_diffusion_model import StableDiffusionFamily
from .stable_diffusion_2.stable_diffusion_2_model import StableDiffusion2Family
from .stable_diffusion_3.stable_diffusion_3_model import (
    SD3T5EncoderModel,
    StableDiffusion3Family,
)
from .stable_diffusion_3_5.stable_diffusion_3_5_model import StableDiffusion3_5Family
from .stable_diffusion_xl.stable_diffusion_xl_model import (
    StableDiffusionXLFamily,
    StableDiffusionXLRefinerFamily,
)

DIFFUSION_FAMILIES = {
    family.model_type: family()
    for family in (
        StableDiffusionFamily,
        StableDiffusion2Family,
        StableDiffusionXLFamily,
        StableDiffusionXLRefinerFamily,
        StableDiffusion3Family,
        StableDiffusion3_5Family,
        QwenImageFamily,
        QwenImage21Family,
    )
}

# Classes that hosted zm_config.json files named before ZMDiffusion replaced the
# per-family container and pipeline classes; their repos load unchanged (same
# model_type, same weights layout).
LEGACY_DIFFUSION_CLASSES = frozenset(
    {
        "ZMDiffusion",
        "StableDiffusionModel",
        "StableDiffusionTextToImage",
        "StableDiffusion2Model",
        "StableDiffusion2TextToImage",
        "StableDiffusionXLModel",
        "StableDiffusionXLTextToImage",
        "StableDiffusionXLRefinerModel",
        "StableDiffusionXLRefinerImageToImage",
        "StableDiffusion3Model",
        "StableDiffusion3TextToImage",
        "StableDiffusion3_5Model",
        "StableDiffusion3_5TextToImage",
        "QwenImageModel",
        "QwenImageTextToImage",
        "QwenImage21Model",
        "QwenImage21TextToImage",
    }
)


def get_diffusion_family(model_type):
    family = DIFFUSION_FAMILIES.get(model_type)
    if family is None:
        raise ValueError(
            f"Unknown diffusion model_type {model_type!r}. ZMDiffusion supports: "
            f"{', '.join(sorted(DIFFUSION_FAMILIES))}."
        )
    return family


class ZMDiffusionMeta(type(BaseModel)):
    """Carries a family config's ``model_type`` into the constructor kwargs.

    ``ZMDiffusion(StableDiffusionXLConfig())`` is expanded by the base metaclass
    into the config's flat fields, which do not include ``model_type`` (a class
    attribute of the config); this adds it so the family can be selected, and keeps
    ``model.config is config`` when nothing overrides it, as for every BaseModel.
    """

    def __call__(cls, *args, **kwargs):
        if not (args and isinstance(args[0], BaseConfig)):
            return super().__call__(*args, **kwargs)
        config, overridden = args[0], bool(kwargs)
        kwargs.setdefault("model_type", config.model_type)
        model = super().__call__(*args, **kwargs)
        if not overridden:
            model.config = config
        return model


@keras.saving.register_keras_serializable(package="zeromodels")
class ZMDiffusion(BaseModel, BaseDiffusion, metaclass=ZMDiffusionMeta):
    """Text-to-image and image-to-image diffusion, pure Keras 3 and cross-backend.

    The one diffusion pipeline class of zeromodels (diffusers' ``DiffusionPipeline``
    in this library's form), for every hosted family: Stable Diffusion 1.x / 2.x /
    XL (base, refiner, Turbo), Stable Diffusion 3 / 3.5, Qwen-Image and
    Qwen-Image-2.1. A checkpoint's ``model_type`` (recorded in its
    ``zm_config.json``) selects the family; the family decides which components
    are built and how a prompt becomes an image, so loading and generating look the
    same for all of them::

        from zeromodels.diffusers import ZMDiffusion, ZMDiffusionTokenizer

        repo = "zeromodels/stable-diffusion-v1-5"
        pipe = ZMDiffusion.from_weights(repo)
        tokenizer = ZMDiffusionTokenizer.from_weights(repo)
        image = pipe.generate(**tokenizer("a photo of a cat"))  # (1, 512, 512, 3) uint8

    One functional ``keras.Model`` holds every trained component as disconnected
    paths (``{"sample", "timestep", "encoder_hidden_states", ...} -> "noise_pred"``
    for the denoiser, ``{"image", "latent"} -> "moments" / "image"`` for the VAE,
    the token ids to the prompt embeddings for the text towers), and exposes them as
    attributes (``.unet`` or ``.transformer``, ``.vae``, ``.text_encoder``,
    ``.text_encoder_2``) that ``generate`` runs step by step. The weights are
    converted once from diffusers checkpoints and hosted under
    ``zeromodels/<variant>``; on-the-fly ``hf:`` conversion is not supported.

    ``generate`` (:class:`BaseDiffusion`) takes the tokenizer's output plus
    ``negative_input_ids``, ``num_inference_steps``, ``guidance_scale``, ``seed``,
    ``latents`` (explicit initial noise, for results identical across backends) and
    ``output_type`` (``"image"`` uint8 or ``"latent"``). Image-to-image takes an
    ``image`` and a ``strength``; ``denoising_end`` / ``denoising_start`` split a run
    between the SDXL base and refiner. Family-specific arguments: SDXL's
    ``original_size`` / ``crops_coords_top_left`` / ``target_size`` (and
    ``aesthetic_score`` on the refiner), SD 3's ``input_ids_3`` (the T5 tokens), and
    Qwen-Image's ``height`` / ``width`` / ``negative_attention_mask``. Defaults come
    from the repo's ``generate_args``.

    SD 3 / 3.5 can attach the separately hosted T5-XXL encoder:
    ``ZMDiffusion.from_weights(repo, text_encoder_3="zeromodels/t5-v1_1-xxl-encoder")``
    or ``pipe.text_encoder_3 = SD3T5EncoderModel.from_weights(...)``; it lives
    outside the checkpoint's weights. Without it the T5 features are zeros.

    Args:
        model_type: The family, e.g. ``"stable_diffusion"``,
            ``"stable_diffusion_xl"``, ``"stable_diffusion_3_5"``, ``"qwen_image"``.
            Read from the repo on ``from_weights``.
        scheduler: A :class:`BaseScheduler`. Defaults to the one the config's
            ``scheduler_config`` describes (the checkpoint's own sampler), else the
            family's default. Swap any time with ``pipe.scheduler = ...``.
        **kwargs: The family config's flat fields (``unet_`` / ``transformer_`` /
            ``vae_`` / ``text_`` prefixed), or pass a family config positionally:
            ``ZMDiffusion(StableDiffusionXLConfig())``. Pass ``unet_sample_size`` /
            ``vae_sample_size`` to build for another resolution.
    """

    # one class serves every family, so the typed config is the family's own
    # (resolved from model_type in __init__ and _config_from_zm_spec), not a class
    # attribute the base metaclass would apply
    config_class = None
    HF_MODEL_TYPE = None
    HUB_REPO_SIBLINGS = LEGACY_DIFFUSION_CLASSES

    def __init__(self, model_type=None, scheduler=None, name="ZMDiffusion", **kwargs):
        family = get_diffusion_family(model_type)
        keras_kwargs = {k: kwargs.pop(k) for k in ("trainable", "dtype") if k in kwargs}
        config = family.config_class.from_dict(kwargs)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        components = family.build_components(config, data_format, channels_axis)
        inputs, outputs = family.build_graph(config, components, data_format)
        super().__init__(inputs=inputs, outputs=outputs, name=name, **keras_kwargs)

        self.model_type = model_type
        self.family = family
        self.config = config
        self.data_format = data_format
        self.channels_axis = channels_axis
        for attr, component in components.items():
            setattr(self, attr, component)
        self.generate_args = dict(getattr(family, "generate_args", None) or {})
        if scheduler is None:
            scheduler = (
                get_scheduler(config.scheduler_config)
                if getattr(config, "scheduler_config", None)
                else family.default_scheduler()
            )
        self.scheduler = scheduler

    @classmethod
    def from_weights(cls, identifier, text_encoder_3=None, **kwargs):
        model = super().from_weights(identifier, **kwargs)
        if text_encoder_3 is not None:
            if not getattr(model.family, "supports_text_encoder_3", False):
                raise ValueError(
                    f"text_encoder_3 is only used by Stable Diffusion 3 / 3.5, not "
                    f"by a {model.model_type!r} checkpoint."
                )
            if isinstance(text_encoder_3, str):
                text_encoder_3 = SD3T5EncoderModel.from_weights(
                    text_encoder_3, load_dtype=kwargs.get("load_dtype")
                )
            model.text_encoder_3 = text_encoder_3
        return model

    @classmethod
    def from_variant(cls, variant, /, quantization=None, **kwargs):
        # a bare name is the hosted repo's basename; `quantization` is applied by
        # from_weights after the load, exactly as on the org/repo branch
        return cls.from_hub_repo(f"zeromodels/{variant}", **kwargs)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        raise NotImplementedError(
            "On-the-fly `hf:` conversion is not supported for diffusion models. Load "
            "the preconverted weights from a zeromodels repo, e.g. "
            "ZMDiffusion.from_weights('zeromodels/stable-diffusion-v1-5'), or run the "
            f"family's convert_*_diffusers_to_keras.py to produce them from {repo!r}."
        )

    @classmethod
    def _config_from_zm_spec(cls, spec, variant):
        # parse the repo's zm_config.json with its family's typed config (refusing
        # keys this version does not know, as the base loader does for typed
        # configs) and keep the model_type the flat kwargs would otherwise lose
        if spec is None or isinstance(spec.get("config"), dict):
            return super()._config_from_zm_spec(spec, variant)
        from zeromodels.conversion.zm_config import ZM_METADATA_KEYS

        fields = {k: v for k, v in spec.items() if k not in ZM_METADATA_KEYS}
        model_type = fields.get("model_type")
        config_cls = get_diffusion_family(model_type).config_class
        unknown = config_cls.unknown_keys(fields)
        if unknown:
            raise ValueError(
                f"ZMDiffusion.from_weights: '{variant}' zm_config.json has config "
                f"key(s) {sorted(unknown)} that this zeromodels version does not "
                f"recognize. The repo is likely newer than the installed zeromodels: "
                f"upgrade it, or remove the key(s) if you are hand-editing the config."
            )
        kwargs = config_cls.from_dict(fields).constructor_kwargs()
        return {"model_type": model_type, **kwargs}

    @staticmethod
    def _apply_generate_args(model, spec):
        # merged over the family's defaults (an instance attribute here, since one
        # class serves every family)
        if spec is not None and spec.get("generate_args"):
            model.generate_args = {**model.generate_args, **spec["generate_args"]}

    def get_config(self):
        config = strip_functional_graph_keys(super().get_config())
        config.update(self.config.constructor_kwargs())
        config["model_type"] = self.model_type
        return config

    @property
    def text_encoder_3(self):
        return self.__dict__.get("text_encoder_3")

    def __setattr__(self, name, value):
        # the T5 encoder is kept out of the tracked (saved / loaded / device-moved)
        # sub-layers: Keras (and torch's Module on that backend) would otherwise
        # register any model assigned as an attribute
        if name == "text_encoder_3":
            self.__dict__["text_encoder_3"] = value
            return
        super().__setattr__(name, value)

    @property
    def latent_shape(self):
        return self.family.latent_shape(self)

    def default_scheduler(self):
        return self.family.default_scheduler()

    def unconditional_ids(self, batch):
        return self.family.unconditional_ids(self, batch)

    def encode_prompt(self, input_ids, attention_mask=None, **conditioning):
        return self.family.encode_prompt(
            self, input_ids, attention_mask, **conditioning
        )

    def family_hook(self, name):
        # the family's own version of a BaseDiffusion hook, bound to this model, or
        # None when the family keeps the shared implementation
        hook = getattr(self.family, name, None)
        return None if hook is None else functools.partial(hook, self)

    def encode_negative_prompt(self, negative_input_ids, batch, **conditioning):
        hook = (
            self.family_hook("encode_negative_prompt") or super().encode_negative_prompt
        )
        return hook(negative_input_ids, batch, **conditioning)

    def predict_noise(self, latents, timesteps, embeddings):
        return self.family.predict_noise(self, latents, timesteps, embeddings)

    def decode_latents(self, latents, **kwargs):
        return self.family.decode_latents(self, latents, **kwargs)

    def encode_latents(self, image):
        hook = self.family_hook("encode_latents")
        if hook is None:
            raise NotImplementedError(
                f"{self.model_type!r} checkpoints do not support image-to-image."
            )
        return hook(image)

    def prepare_latents(self, batch, seed=None, latents=None, dtype="float32"):
        hook = self.family_hook("prepare_latents") or super().prepare_latents
        return hook(batch, seed=seed, latents=latents, dtype=dtype)

    def denoise(
        self, latents, embeddings, num_inference_steps, guidance_scale, timesteps=None
    ):
        hook = self.family_hook("denoise") or super().denoise
        return hook(
            latents,
            embeddings,
            num_inference_steps,
            guidance_scale,
            timesteps=timesteps,
        )

    def generate(self, input_ids, **kwargs):
        hook = self.family_hook("generate") or super().generate
        return hook(input_ids, **kwargs)
