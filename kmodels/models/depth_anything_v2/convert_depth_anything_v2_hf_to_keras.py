import numpy as np

from kmodels.models.depth_anything_v1.convert_depth_anything_v1_hf_to_keras import (
    transfer_depth_anything_v1_weights,
)

if __name__ == "__main__":
    import gc

    import keras
    import torch
    from transformers import DepthAnythingForDepthEstimation

    from kmodels.models.depth_anything_v2.config import (
        DEPTH_ANYTHING_V2_MODEL_CONFIG,
    )
    from kmodels.models.depth_anything_v2.depth_anything_v2_model import (
        DepthAnythingV2Base,
        DepthAnythingV2Large,
        DepthAnythingV2MetricIndoorBase,
        DepthAnythingV2MetricIndoorLarge,
        DepthAnythingV2MetricIndoorSmall,
        DepthAnythingV2MetricOutdoorBase,
        DepthAnythingV2MetricOutdoorLarge,
        DepthAnythingV2MetricOutdoorSmall,
        DepthAnythingV2Small,
    )

    DEPTH_ANYTHING_V2_HF_MODEL_IDS = {
        "DepthAnythingV2Small": "depth-anything/Depth-Anything-V2-Small-hf",
        "DepthAnythingV2Base": "depth-anything/Depth-Anything-V2-Base-hf",
        "DepthAnythingV2Large": "depth-anything/Depth-Anything-V2-Large-hf",
        "DepthAnythingV2MetricIndoorSmall": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "DepthAnythingV2MetricIndoorBase": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        "DepthAnythingV2MetricIndoorLarge": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "DepthAnythingV2MetricOutdoorSmall": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "DepthAnythingV2MetricOutdoorBase": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
        "DepthAnythingV2MetricOutdoorLarge": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    }

    VARIANTS = [
        ("DepthAnythingV2Small", DepthAnythingV2Small, "depth_anything_v2_small"),
        ("DepthAnythingV2Base", DepthAnythingV2Base, "depth_anything_v2_base"),
        ("DepthAnythingV2Large", DepthAnythingV2Large, "depth_anything_v2_large"),
        (
            "DepthAnythingV2MetricIndoorSmall",
            DepthAnythingV2MetricIndoorSmall,
            "depth_anything_v2_metric_indoor_small",
        ),
        (
            "DepthAnythingV2MetricIndoorBase",
            DepthAnythingV2MetricIndoorBase,
            "depth_anything_v2_metric_indoor_base",
        ),
        (
            "DepthAnythingV2MetricIndoorLarge",
            DepthAnythingV2MetricIndoorLarge,
            "depth_anything_v2_metric_indoor_large",
        ),
        (
            "DepthAnythingV2MetricOutdoorSmall",
            DepthAnythingV2MetricOutdoorSmall,
            "depth_anything_v2_metric_outdoor_small",
        ),
        (
            "DepthAnythingV2MetricOutdoorBase",
            DepthAnythingV2MetricOutdoorBase,
            "depth_anything_v2_metric_outdoor_base",
        ),
        (
            "DepthAnythingV2MetricOutdoorLarge",
            DepthAnythingV2MetricOutdoorLarge,
            "depth_anything_v2_metric_outdoor_large",
        ),
    ]

    for name, ctor, save_name in VARIANTS:
        hf_id = DEPTH_ANYTHING_V2_HF_MODEL_IDS[name]
        config = DEPTH_ANYTHING_V2_MODEL_CONFIG[name]
        print(f"\n{'=' * 60}")
        print(f"Converting: {name}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = DepthAnythingForDepthEstimation.from_pretrained(hf_id).eval()
        hf_sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model = ctor(input_shape=(518, 518, 3), weights=None)
        transfer_depth_anything_v1_weights(keras_model, hf_sd, config)

        np.random.seed(42)
        test_image = np.random.rand(1, 518, 518, 3).astype(np.float32)

        keras_output = keras_model.predict(test_image, verbose=0)
        keras_depth = keras_output

        with torch.no_grad():
            hf_input = torch.from_numpy(test_image.transpose(0, 3, 1, 2))
            hf_output = hf_model(pixel_values=hf_input)
            hf_depth = hf_output.predicted_depth.cpu().numpy()

        keras_depth_squeezed = keras_depth.squeeze(-1)

        depth_diff = float(np.max(np.abs(keras_depth_squeezed - hf_depth)))
        mean_diff = float(np.mean(np.abs(keras_depth_squeezed - hf_depth)))
        print(f"  Max depth diff: {depth_diff:.6f}")
        print(f"  Mean depth diff: {mean_diff:.6f}")
        assert depth_diff < 1.0, f"{name}: depth diff {depth_diff:.2e}"
        print("  Verification OK")

        out = f"{save_name}.weights.h5"
        keras_model.save_weights(out)
        print(f"  Saved -> {out}")

        del keras_model, hf_model, hf_sd
        keras.backend.clear_session()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
