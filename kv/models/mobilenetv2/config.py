MOBILENETV2_MODEL_CONFIG = {
    "MobileNetV2WM50": {
        "width_multiplier": 0.5,
        "depth_multiplier": 1.0,
        "fix_channels": False,
    },
    "MobileNetV2WM100": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "fix_channels": False,
    },
    "MobileNetV2WM110": {
        "width_multiplier": 1.1,
        "depth_multiplier": 1.2,
        "fix_channels": True,
    },
    "MobileNetV2WM120": {
        "width_multiplier": 1.2,
        "depth_multiplier": 1.4,
        "fix_channels": True,
    },
    "MobileNetV2WM140": {
        "width_multiplier": 1.4,
        "depth_multiplier": 1.0,
        "fix_channels": False,
    },
}

MOBILENETV2_WEIGHTS_CONFIG = {
    "MobileNetV2WM50": {
        "lamb_in1k": {"url": ""},
    },
    "MobileNetV2WM100": {
        "ra_in1k": {"url": ""},
    },
    "MobileNetV2WM110": {
        "ra_in1k": {"url": ""},
    },
    "MobileNetV2WM120": {
        "ra_in1k": {"url": ""},
    },
    "MobileNetV2WM140": {
        "ra_in1k": {"url": ""},
    },
}
