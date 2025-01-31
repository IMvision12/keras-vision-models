MOBILENETV2_MODEL_CONFIG = {
    "MobileNetV2WM50": {
        "width_multiplier": 0.5,
        "depth_multiplier": 1.0,
    },
    "MobileNetV2WM100": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
    },
    "MobileNetV2WM140": {
        "width_multiplier": 1.4,
        "depth_multiplier": 1.0,
    },
}

MOBILENETV2_WEIGHTS_CONFIG = {
    "MobileNetV2WM50": {
        "lamb_in1k": {"url": ""},
    },
    "MobileNetV2WM100": {
        "ra_in1k": {"url": ""},
    },
    "MobileNetV2WM140": {
        "ra_in1k": {"url": ""},
    },
}
