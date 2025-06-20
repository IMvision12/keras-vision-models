import keras
from kvmm.models.vlms.siglip.siglip_model import SigLIPModel

from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config