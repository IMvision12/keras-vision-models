# Import internal components (for internal use)
# Import model variants
from . import yolov5
from .blocks import bottleneck_block, c3_block, conv_block, sppf_block
from .head import detect_head
from .layers import DFL
from .utils import (
    decode_bboxes,
    dist2bbox,
    make_anchors,
    make_divisible,
    scale_channels,
    scale_depth,
)

# For backward compatibility, also expose YoloV5s directly
from .yolov5 import YoloV5s
