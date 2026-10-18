from zeromodels.diffusers.stable_diffusion_3.stable_diffusion_3_model import (
    StableDiffusion3Family,
)

from .stable_diffusion_3_5_config import StableDiffusion3_5Config


class StableDiffusion3_5Family(StableDiffusion3Family):
    """Stable Diffusion 3.5 inside :class:`ZMDiffusion` (``model_type="stable_diffusion_3_5"``).

    The SD 3 family with the SD 3.5 configuration (:class:`StableDiffusion3_5Config`):
    RMS-normalized queries and keys in every MMDiT block and, for the medium
    checkpoint, the dual-attention (MMDiT-X) blocks. Same components
    (``transformer`` / ``vae`` / ``text_encoder`` / ``text_encoder_2``), same
    separately hosted T5-XXL (``text_encoder_3``). Each repo's ``generate_args`` hold
    the checkpoint's defaults: 28 steps at guidance 3.5 for large, 40 at 4.5 for
    medium, 4 without guidance for large-turbo.
    """

    model_type = "stable_diffusion_3_5"
    config_class = StableDiffusion3_5Config
    generate_args = {"num_inference_steps": 28, "guidance_scale": 3.5}
