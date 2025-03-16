MOBILENETV3_MODEL_CONFIG = {
    "MobileNetV3Small075": {
        "width_multiplier": 0.75,
        "depth_multiplier": 1.0,
        "config": "small",
        "minimal": False,
    },
    "MobileNetV3Small100": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "small",
        "minimal": False,
    },
    "MobileNetV3SmallMinimal100": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "small",
        "minimal": True,
    },
    "MobileNetV3Large075": {
        "width_multiplier": 0.75,
        "depth_multiplier": 1.0,
        "config": "large",
        "minimal": False,
    },
    "MobileNetV3Large100": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "large",
        "minimal": False,
    },
    "MobileNetV3LargeMinimal100": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "large",
        "minimal": True,
    },
}


MOBILENETV3_WEIGHTS_CONFIG = {
    "MobileNetV3Small075": {
        "imagenet": {"url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/keras_org_mobilenetv3small075.keras"},
    },
    "MobileNetV3Small100": {
        "imagenet": {"url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/keras_org_mobilenetv3small100.keras"},
    },
    "MobileNetV3SmallMinimal100": {
        "imagenet": {"url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/keras_org_mobilenetv3smallminimal100.keras"},
    },
    "MobileNetV3Large075": {
        "imagenet": {"url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/keras_org_mobilenetv3large075.keras"},
    },
    "MobileNetV3Large100": {
        "imagenet": {"url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/keras_org_mobilenetv3large100.keras"},
    },
    "MobileNetV3LargeMinimal100": {
        "imagenet": {"url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/keras_org_mobilenetv3largeminimal100.keras"},
    },
}
