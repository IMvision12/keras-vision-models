_V2_SMALL_BASE = {
    "backbone_dim": 384,
    "backbone_depth": 12,
    "backbone_num_heads": 6,
    "out_indices": [3, 6, 9, 12],
    "neck_hidden_sizes": [48, 96, 192, 384],
    "fusion_hidden_size": 64,
    "reassemble_factors": [4, 2, 1, 0.5],
}

_V2_BASE_BASE = {
    "backbone_dim": 768,
    "backbone_depth": 12,
    "backbone_num_heads": 12,
    "out_indices": [3, 6, 9, 12],
    "neck_hidden_sizes": [96, 192, 384, 768],
    "fusion_hidden_size": 128,
    "reassemble_factors": [4, 2, 1, 0.5],
}

_V2_LARGE_BASE = {
    "backbone_dim": 1024,
    "backbone_depth": 24,
    "backbone_num_heads": 16,
    "out_indices": [5, 12, 18, 24],
    "neck_hidden_sizes": [256, 512, 1024, 1024],
    "fusion_hidden_size": 256,
    "reassemble_factors": [4, 2, 1, 0.5],
}

DEPTH_ANYTHING_V2_MODEL_CONFIG = {
    "DepthAnythingV2Small": {**_V2_SMALL_BASE},
    "DepthAnythingV2Base": {**_V2_BASE_BASE},
    "DepthAnythingV2Large": {**_V2_LARGE_BASE},
    "DepthAnythingV2MetricIndoorSmall": {
        **_V2_SMALL_BASE,
        "depth_estimation_type": "metric",
        "max_depth": 20.0,
    },
    "DepthAnythingV2MetricIndoorBase": {
        **_V2_BASE_BASE,
        "depth_estimation_type": "metric",
        "max_depth": 20.0,
    },
    "DepthAnythingV2MetricIndoorLarge": {
        **_V2_LARGE_BASE,
        "depth_estimation_type": "metric",
        "max_depth": 20.0,
    },
    "DepthAnythingV2MetricOutdoorSmall": {
        **_V2_SMALL_BASE,
        "depth_estimation_type": "metric",
        "max_depth": 80.0,
    },
    "DepthAnythingV2MetricOutdoorBase": {
        **_V2_BASE_BASE,
        "depth_estimation_type": "metric",
        "max_depth": 80.0,
    },
    "DepthAnythingV2MetricOutdoorLarge": {
        **_V2_LARGE_BASE,
        "depth_estimation_type": "metric",
        "max_depth": 80.0,
    },
}

DEPTH_ANYTHING_V2_WEIGHTS_CONFIG = {
    "DepthAnythingV2Small": {
        "da_v2": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/depth_anything_v2/depth_anything_v2_small.weights.h5",
        },
    },
    "DepthAnythingV2Base": {
        "da_v2": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/depth_anything_v2/depth_anything_v2_base.weights.h5",
        },
    },
    "DepthAnythingV2Large": {
        "da_v2": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/depth_anything_v2/depth_anything_v2_large.weights.h5",
        },
    },
    "DepthAnythingV2MetricIndoorSmall": {
        "da_v2": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/depth_anything_v2/depth_anything_v2_metric_indoor_small.weights.h5",
        },
    },
    "DepthAnythingV2MetricIndoorBase": {
        "da_v2": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/depth_anything_v2/depth_anything_v2_metric_indoor_base.weights.h5",
        },
    },
    "DepthAnythingV2MetricIndoorLarge": {
        "da_v2": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/depth_anything_v2/depth_anything_v2_metric_indoor_large.weights.h5",
        },
    },
    "DepthAnythingV2MetricOutdoorSmall": {
        "da_v2": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/depth_anything_v2/depth_anything_v2_metric_outdoor_small.weights.h5",
        },
    },
    "DepthAnythingV2MetricOutdoorBase": {
        "da_v2": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/depth_anything_v2/depth_anything_v2_metric_outdoor_base.weights.h5",
        },
    },
    "DepthAnythingV2MetricOutdoorLarge": {
        "da_v2": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/depth_anything_v2/depth_anything_v2_metric_outdoor_large.weights.h5",
        },
    },
}

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
