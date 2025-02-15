RESNET_MODEL_CONFIG = {
    "ResNet50": {
        "block_fn": "bottleneck_block",
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
    },
    "ResNet101": {
        "block_fn": "bottleneck_block",
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
    },
    "ResNet152": {
        "block_fn": "bottleneck_block",
        "block_repeats": [3, 8, 36, 3],
        "filters": [64, 128, 256, 512],
    },
}

RESNET_WEIGHTS_CONFIG = {
    # ResNet Variants
    "ResNet50": {
        "tv_in1k": {
            "url": "",
        },
        "a1_in1k": {
            "url": "",
        },
        "gluon_in1k": {
            "url": "",
        },
    },
    "ResNet101": {
        "tv_in1k": {
            "url": "",
        },
        "a1_in1k": {
            "url": "",
        },
        "gluon_in1k": {
            "url": "",
        },
    },
    "ResNet152": {
        "tv_in1k": {
            "url": "",
        },
        "a1_in1k": {
            "url": "",
        },
        "gluon_in1k": {
            "url": "",
        },
    },
}
