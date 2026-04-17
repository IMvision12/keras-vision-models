"""Parity harness for processor data_format wiring.

Captures every processor's output into a pickle and then compares two
pickles. Modes:

    python parity_harness.py capture BEFORE <pickle>            # no kwargs
    python parity_harness.py capture default <pickle>           # AFTER default
    python parity_harness.py capture cf <pickle>                # AFTER cf
    python parity_harness.py compare <a_pickle> <b_pickle>      # diffs
    python parity_harness.py compare-transpose <cl> <cf>        # cf vs cl-transposed

The first form calls every processor WITHOUT a ``data_format`` kwarg, so
it works against both pre-wiring and post-wiring code (the pre-wiring
code doesn't accept that kwarg yet).
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from keras import ops

ASSET = "assets/coco_horse_dog.jpg"


def _to_np(x):
    if x is None:
        return None
    if isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (tuple, list)):
        return [_to_np(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_np(v) for k, v in x.items()}
    try:
        arr = ops.convert_to_numpy(x)
        if isinstance(arr, np.generic):
            return arr.item()
        return arr
    except Exception:
        return x


def _maybe_kw(data_format):
    return {} if data_format is None else {"data_format": data_format}


def run_depth_anything_v1(data_format):
    from kmodels.models.depth_anything_v1 import DepthAnythingV1ImageProcessor

    return DepthAnythingV1ImageProcessor(ASSET, **_maybe_kw(data_format))


def run_depth_anything_v2(data_format):
    from kmodels.models.depth_anything_v2 import DepthAnythingV2ImageProcessor

    return DepthAnythingV2ImageProcessor(ASSET, **_maybe_kw(data_format))


def run_detr(data_format):
    from kmodels.models.detr import DETRImageProcessor

    return DETRImageProcessor(ASSET, **_maybe_kw(data_format))


def run_rt_detr(data_format):
    from kmodels.models.rt_detr import RTDETRImageProcessor

    return RTDETRImageProcessor(ASSET, **_maybe_kw(data_format))


def run_rt_detr_v2(data_format):
    from kmodels.models.rt_detr_v2 import RTDETRV2ImageProcessor

    return RTDETRV2ImageProcessor(ASSET, **_maybe_kw(data_format))


def run_rf_detr(data_format):
    from kmodels.models.rf_detr import RFDETRImageProcessor

    return RFDETRImageProcessor(ASSET, **_maybe_kw(data_format))


def run_dfine(data_format):
    from kmodels.models.dfine.dfine_image_processor import DFineImageProcessor

    return DFineImageProcessor(ASSET, **_maybe_kw(data_format))


def run_segformer(data_format):
    from kmodels.models.segformer.segformer_image_preprocessor import (
        SegFormerImageProcessor,
    )

    return SegFormerImageProcessor(ASSET, **_maybe_kw(data_format))


def run_deeplabv3(data_format):
    from kmodels.models.deeplabv3.deeplabv3_image_processor import (
        DeepLabV3ImageProcessor,
    )

    return DeepLabV3ImageProcessor(ASSET, **_maybe_kw(data_format))


def run_eomt(data_format):
    from kmodels.models.eomt.eomt_image_processor import EoMTImageProcessor

    return EoMTImageProcessor(ASSET, **_maybe_kw(data_format))


def run_sam(data_format):
    from kmodels.models.sam import SAMImageProcessor

    return SAMImageProcessor(ASSET, **_maybe_kw(data_format))


def run_sam2(data_format):
    from kmodels.models.sam2 import Sam2ImageProcessor

    return Sam2ImageProcessor(ASSET, **_maybe_kw(data_format))


def run_clip(data_format):
    from kmodels.models.clip.clip_image_processor import CLIPImageProcessor

    kw = {} if data_format is None else {"data_format": data_format}
    processor = CLIPImageProcessor(**kw)
    return processor(image_paths=ASSET)


def run_siglip(data_format):
    from kmodels.models.siglip.siglip_image_processor import SigLIPImageProcessor

    kw = {} if data_format is None else {"data_format": data_format}
    processor = SigLIPImageProcessor(**kw)
    return processor(image_paths=ASSET)


def run_siglip2(data_format):
    from kmodels.models.siglip2.siglip2_image_processor import (
        SigLIP2ImageProcessor,
    )

    kw = {} if data_format is None else {"data_format": data_format}
    processor = SigLIP2ImageProcessor(**kw)
    return processor(image_paths=ASSET)


PROCESSORS = {
    "depth_anything_v1": run_depth_anything_v1,
    "depth_anything_v2": run_depth_anything_v2,
    "detr": run_detr,
    "rt_detr": run_rt_detr,
    "rt_detr_v2": run_rt_detr_v2,
    "rf_detr": run_rf_detr,
    "dfine": run_dfine,
    "segformer": run_segformer,
    "deeplabv3": run_deeplabv3,
    "eomt": run_eomt,
    "sam": run_sam,
    "sam2": run_sam2,
    "clip": run_clip,
    "siglip": run_siglip,
    "siglip2": run_siglip2,
}


def capture(mode, out_path):
    if mode == "BEFORE":
        data_format = None
    elif mode == "default":
        data_format = None  # let processor resolve via global config
    elif mode == "cf":
        data_format = "channels_first"
    elif mode == "cl":
        data_format = "channels_last"
    else:
        raise SystemExit(f"bad mode: {mode}")

    results = {}
    for name, fn in PROCESSORS.items():
        print(f"[{mode}] running {name} ...", flush=True)
        try:
            out = fn(data_format)
            results[name] = _to_np(out)
        except Exception as exc:
            traceback.print_exc()
            results[name] = {"__error__": f"{type(exc).__name__}: {exc}"}

    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"wrote {out_path}", flush=True)


def _walk(prefix, a, b, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a.keys()) | set(b.keys()))
        for k in keys:
            _walk(f"{prefix}.{k}" if prefix else str(k), a.get(k), b.get(k), diffs)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(
                (prefix, float("nan"), False, f"list-len {len(a)} vs {len(b)}")
            )
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(f"{prefix}[{i}]", x, y, diffs)
        return
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            diffs.append((prefix, float("nan"), False, f"shape {a.shape} vs {b.shape}"))
            return
        if a.size == 0:
            diffs.append((prefix, 0.0, True, "empty"))
            return
        af = a.astype(np.float64)
        bf = b.astype(np.float64)
        d = float(np.max(np.abs(af - bf)))
        diffs.append((prefix, d, True, f"shape={a.shape}"))
        return
    if a == b:
        diffs.append((prefix, 0.0, True, f"scalar={a!r}"))
    else:
        diffs.append((prefix, float("nan"), False, f"{a!r} vs {b!r}"))


def compare(a_path, b_path):
    with open(a_path, "rb") as f:
        a = pickle.load(f)
    with open(b_path, "rb") as f:
        b = pickle.load(f)
    print(f"{'processor':<22} {'max|diff|':>14}  status")
    print("-" * 60)
    worst_per_proc = {}
    for name in sorted(set(a.keys()) | set(b.keys())):
        diffs = []
        _walk("", a.get(name), b.get(name), diffs)
        if not diffs:
            worst_per_proc[name] = ("", 0.0, True, "no fields")
            continue
        worst = max(
            diffs,
            key=lambda t: (
                0 if t[2] and not np.isnan(t[1]) else 1,
                -t[1] if np.isnan(t[1]) else t[1],
            ),
        )
        worst_per_proc[name] = worst
        status = "OK" if worst[2] else "SHAPE/TYPE MISMATCH"
        val = "NaN" if np.isnan(worst[1]) else f"{worst[1]:.3e}"
        print(f"{name:<22} {val:>14}  {status}  [{worst[0]}]")
    n_ok = sum(1 for w in worst_per_proc.values() if w[2])
    print(f"\n{n_ok}/{len(worst_per_proc)} processors pass")


def compare_transpose(cl_path, cf_path):
    """Compare channels_first output against transposed channels_last output.

    Walks both pickles. When both values are 4-D ndarrays with matching batch
    size but different channel axis, transpose cf from (B,C,H,W) back to
    (B,H,W,C) and diff with cl. Non-tensor fields are compared as-is.
    """
    with open(cl_path, "rb") as f:
        cl = pickle.load(f)
    with open(cf_path, "rb") as f:
        cf = pickle.load(f)
    print(f"{'processor':<22} {'max|diff|':>14}  status")
    print("-" * 60)

    for name in sorted(set(cl.keys()) | set(cf.keys())):
        diffs = []
        _walk_transpose("", cl.get(name), cf.get(name), diffs)
        if not diffs:
            print(f"{name:<22} {'-':>14}  no fields")
            continue
        worst = max(
            diffs,
            key=lambda t: (
                0 if t[2] and not np.isnan(t[1]) else 1,
                -t[1] if np.isnan(t[1]) else t[1],
            ),
        )
        status = "OK" if worst[2] else "SHAPE/TYPE MISMATCH"
        val = "NaN" if np.isnan(worst[1]) else f"{worst[1]:.3e}"
        print(f"{name:<22} {val:>14}  {status}  [{worst[0]}]")


def _walk_transpose(prefix, a, b, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a.keys()) | set(b.keys()))
        for k in keys:
            _walk_transpose(
                f"{prefix}.{k}" if prefix else str(k), a.get(k), b.get(k), diffs
            )
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(
                (prefix, float("nan"), False, f"list-len {len(a)} vs {len(b)}")
            )
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk_transpose(f"{prefix}[{i}]", x, y, diffs)
        return
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape == b.shape:
            # non-image tensor or both stored in same layout
            if a.size == 0:
                diffs.append((prefix, 0.0, True, "empty"))
                return
            d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            diffs.append((prefix, d, True, f"same-shape {a.shape}"))
            return
        # 4-D case: try (B,H,W,C) vs (B,C,H,W)
        if a.ndim == 4 and b.ndim == 4 and a.shape[0] == b.shape[0]:
            # transpose b from (B,C,H,W) to (B,H,W,C)
            try:
                bt = np.transpose(b, (0, 2, 3, 1))
                if bt.shape == a.shape:
                    d = float(
                        np.max(np.abs(a.astype(np.float64) - bt.astype(np.float64)))
                    )
                    diffs.append((prefix, d, True, f"cl vs cf-transposed {a.shape}"))
                    return
            except Exception:
                pass
        diffs.append(
            (prefix, float("nan"), False, f"shape mismatch {a.shape} vs {b.shape}")
        )
        return
    if a == b:
        diffs.append((prefix, 0.0, True, f"scalar={a!r}"))
    else:
        diffs.append((prefix, float("nan"), False, f"{a!r} vs {b!r}"))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "capture":
        capture(sys.argv[2], sys.argv[3])
    elif cmd == "compare":
        compare(sys.argv[2], sys.argv[3])
    elif cmd == "compare-transpose":
        compare_transpose(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
