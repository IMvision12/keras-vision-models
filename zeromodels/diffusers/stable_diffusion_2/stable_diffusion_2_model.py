from zeromodels.diffusers.stable_diffusion.stable_diffusion_model import (
    StableDiffusionFamily,
)

from .stable_diffusion_2_config import StableDiffusion2Config


class StableDiffusion2Family(StableDiffusionFamily):
    """Stable Diffusion 2.x inside :class:`ZMDiffusion` (``model_type="stable_diffusion_2"``).

    The SD 1.x family with the SD 2.x configuration (:class:`StableDiffusion2Config`):
    the same three components and paths, with a 1024-d text context, one attention
    head count per UNet level, a linear token projection in the Transformer2D blocks
    and the 23-layer ``gelu`` OpenCLIP ViT-H/14 text tower. The checkpoint's
    ``!``-padded empty prompt drives classifier-free guidance and its sampler (PNDM
    for ``stable-diffusion-2-1-base``, DDIM for the others, v-prediction for the
    768px checkpoints) comes from the repo's ``scheduler_config``. The 768px repos
    already carry ``unet_sample_size=96, vae_sample_size=768``.
    """

    model_type = "stable_diffusion_2"
    config_class = StableDiffusion2Config
