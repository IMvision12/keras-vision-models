RESNET_MODEL_CONFIG = {
    "ResNet50": {
        "block_fn": "bottleneck_block",
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
    },
    "ResNet101": {
        "block_fn": "bottleneck",
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
    },
    "ResNet152": {
        "block_fn": "bottleneck",
        "block_repeats": [3, 8, 36, 3],
        "filters": [64, 128, 256, 512],
    },
}

RESNET_WEIGHTS_CONFIG = {
    # ResNet Variants
    "ResNet50": {
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_tv_in1k.keras",
        },
        "a1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
    },
    "ResNet101": {
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet101_tv_in1k.keras",
        },
        "a1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet101_a1_in1k.keras",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
    },
    "ResNet152": {
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet152_tv_in1k.keras",
        },
        "a1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet152_a1_in1k.keras",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
    },
}
