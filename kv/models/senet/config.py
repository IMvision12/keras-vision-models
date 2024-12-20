SENET_MODEL_CONFIG = {
    "SEResNet50": {
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "senet": True,
    },
    "SEResNeXt50_32x4d": {
        "block_fn": "resnext_block",
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "senet": True,
    },
    "SEResNeXt101_32x4d": {
        "block_fn": "resnext_block",
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "senet": True,
    },
    "SEResNeXt101_32x8d": {
        "block_fn": "resnext_block",
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 4,
        "senet": True,
    },
}

SENET_WEIGHTS_CONFIG = {
    # SE-ResNet and SE-ResNeXt Variants
    "SEResNet50": {
        "a1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/seresnet50_a1_in1k.keras",
        }
    },
    "SEResNeXt50_32x4d": {
        "racm_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/seresnext50_32x4d_racm_in1k.keras",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
    },
    "SEResNeXt101_32x4d": {
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
    },
    "SEResNeXt101_32x8d": {
        "ah_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/seresnext101_32x8d_ah_in1k.keras",
        }
    },
}
