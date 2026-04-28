from __future__ import annotations

import os

import keras
import numpy as np
import pytest
from PIL import Image

transformers = pytest.importorskip("transformers")

from transformers import (
    CLIPImageProcessor as HFCLIPImageProcessor,
)
from transformers import (
    DetrImageProcessor,
    DPTImageProcessor,
    EomtImageProcessor,
    RTDetrImageProcessor,
    SamImageProcessor,
    SegformerImageProcessor,
    SiglipImageProcessor,
)
from transformers import (
    Sam2ImageProcessor as HFSam2ImageProcessor,
)

from kmodels.models.clip.clip_image_processor import (
    CLIPImageProcessor as KerasCLIPImageProcessor,
)
from kmodels.models.depth_anything_v1 import DepthAnythingV1ImageProcessor
from kmodels.models.depth_anything_v2 import DepthAnythingV2ImageProcessor
from kmodels.models.detr import DETRImageProcessor
from kmodels.models.dfine.dfine_image_processor import DFineImageProcessor
from kmodels.models.eomt.eomt_image_processor import EoMTImageProcessor
from kmodels.models.metaclip2 import MetaClip2ImageProcessor
from kmodels.models.rt_detr import RTDETRImageProcessor
from kmodels.models.rt_detr_v2 import RTDETRV2ImageProcessor
from kmodels.models.sam import SAMImageProcessor
from kmodels.models.sam2 import Sam2ImageProcessor
from kmodels.models.segformer.segformer_image_preprocessor import (
    SegFormerImageProcessor,
)
from kmodels.models.siglip.siglip_image_processor import (
    SigLIPImageProcessor as KerasSigLIPImageProcessor,
)

ASSET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets", "coco_horse_dog.jpg")
)


def _to_channels_last(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
        return np.transpose(arr, (0, 2, 3, 1))
    return arr


def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = _to_channels_last(a)
    b = _to_channels_last(b)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _pil_image():
    return Image.open(ASSET_PATH).convert("RGB")


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "numpy"):
        return x.numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return keras.ops.convert_to_numpy(x)


def _run_detr(data_format):
    ours = _as_numpy(
        DETRImageProcessor(
            size={"height": 800, "width": 800},
            data_format=data_format,
        )(ASSET_PATH)
    )
    hf = DetrImageProcessor(
        do_resize=True,
        size={"height": 800, "width": 800},
        do_rescale=True,
        do_normalize=True,
        do_pad=False,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_rt_detr(data_format):
    ours = _as_numpy(
        RTDETRImageProcessor(
            size={"height": 640, "width": 640},
            data_format=data_format,
        )(ASSET_PATH)
    )
    hf = RTDetrImageProcessor(
        do_resize=True,
        size={"height": 640, "width": 640},
        do_rescale=True,
        do_normalize=False,
        do_pad=False,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_rt_detr_v2(data_format):
    ours = _as_numpy(
        RTDETRV2ImageProcessor(
            size={"height": 640, "width": 640},
            data_format=data_format,
        )(ASSET_PATH)
    )
    hf = RTDetrImageProcessor(
        do_resize=True,
        size={"height": 640, "width": 640},
        do_rescale=True,
        do_normalize=False,
        do_pad=False,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_dfine(data_format):
    ours = _as_numpy(DFineImageProcessor(data_format=data_format)(ASSET_PATH))
    hf = RTDetrImageProcessor(
        do_resize=True,
        size={"height": 640, "width": 640},
        do_rescale=True,
        do_normalize=False,
        do_pad=False,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_segformer(data_format):
    ours = _as_numpy(SegFormerImageProcessor(data_format=data_format)(ASSET_PATH))
    hf = SegformerImageProcessor(
        do_resize=True,
        size={"height": 512, "width": 512},
        do_rescale=True,
        do_normalize=True,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_eomt(data_format):
    ours = _as_numpy(EoMTImageProcessor(data_format=data_format)(ASSET_PATH))
    hf = EomtImageProcessor(
        do_resize=True,
        size={"longest_edge": 640, "shortest_edge": 640},
        do_pad=True,
        do_rescale=True,
        do_normalize=True,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_sam(data_format):
    ours = _as_numpy(
        SAMImageProcessor(data_format=data_format)(ASSET_PATH)["pixel_values"]
    )
    hf = SamImageProcessor(
        size={"longest_edge": 1024},
        pad_size={"height": 1024, "width": 1024},
        do_rescale=True,
        do_normalize=True,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_sam2(data_format):
    ours = _as_numpy(
        Sam2ImageProcessor(data_format=data_format)(ASSET_PATH)["pixel_values"]
    )
    hf = HFSam2ImageProcessor()(images=_pil_image(), return_tensors="np")[
        "pixel_values"
    ]
    return ours, hf


def _run_clip(data_format):
    processor = KerasCLIPImageProcessor(data_format=data_format)
    ours = _as_numpy(processor(image_paths=ASSET_PATH)["images"])
    hf = HFCLIPImageProcessor()(images=_pil_image(), return_tensors="np")[
        "pixel_values"
    ]
    return ours, hf


def _run_siglip(data_format):
    processor = KerasSigLIPImageProcessor(data_format=data_format)
    ours = _as_numpy(processor(image_paths=ASSET_PATH))
    hf = SiglipImageProcessor()(images=_pil_image(), return_tensors="np")[
        "pixel_values"
    ]
    return ours, hf


def _run_siglip2(data_format):
    pytest.skip(
        "HF Siglip2ImageProcessor is NaFlex-only; our processor emits a "
        "square image tensor. Not directly comparable."
    )


def _run_metaclip2(data_format):
    processor = MetaClip2ImageProcessor(data_format=data_format)
    ours = _as_numpy(processor(image_paths=ASSET_PATH)["images"])
    hf = HFCLIPImageProcessor(
        do_resize=True,
        size={"height": 224, "width": 224},
        do_center_crop=False,
        do_rescale=True,
        do_normalize=True,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_depth_anything_v1(data_format):
    ours = _as_numpy(
        DepthAnythingV1ImageProcessor(data_format=data_format)(ASSET_PATH)[
            "pixel_values"
        ]
    )
    hf = DPTImageProcessor(
        do_resize=True,
        size={"height": 518, "width": 518},
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        do_rescale=True,
        do_normalize=True,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        resample=Image.BICUBIC,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_depth_anything_v2(data_format):
    ours = _as_numpy(
        DepthAnythingV2ImageProcessor(data_format=data_format)(ASSET_PATH)[
            "pixel_values"
        ]
    )
    hf = DPTImageProcessor(
        do_resize=True,
        size={"height": 518, "width": 518},
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        do_rescale=True,
        do_normalize=True,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        resample=Image.BICUBIC,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


PROCESSORS = {
    "detr": (_run_detr, 5e-2),
    "rt_detr": (_run_rt_detr, 1e-2),
    "rt_detr_v2": (_run_rt_detr_v2, 1e-2),
    "dfine": (_run_dfine, 1e-2),
    "segformer": (_run_segformer, 1.0),
    "eomt": (_run_eomt, 1e-5),
    "sam": (_run_sam, 5e-2),
    "sam2": (_run_sam2, 5e-2),
    "clip": (_run_clip, 5e-2),
    "siglip": (_run_siglip, 5e-2),
    "siglip2": (_run_siglip2, 0.0),
    "metaclip2": (_run_metaclip2, 5e-2),
    "depth_anything_v1": (_run_depth_anything_v1, 5e-1),
    "depth_anything_v2": (_run_depth_anything_v2, 5e-1),
}


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
@pytest.mark.parametrize("name", list(PROCESSORS.keys()))
def test_processor_hf_parity(name, data_format):
    runner, threshold = PROCESSORS[name]
    ours, hf = runner(data_format)
    diff = _max_diff(ours, hf)
    assert diff < threshold, (
        f"{name}[{data_format}] max|diff|={diff:.3e} exceeds {threshold:.1e}"
    )
    print(f"[{name:<20}] {data_format:<15} max|diff|={diff:.3e}")
