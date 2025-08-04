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
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resnet50_tv_in1k.weights.h5",
        },
        "a1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resnet50_a1_in1k.weights.h5",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resnet50_gluon_in1k.weights.h5",
        },
    },
    "ResNet101": {
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resnet101_tv_in1k.weights.h5",
        },
        "a1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resnet101_a1_in1k.weights.h5",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resnet101_gluon_in1k.weights.h5",
        },
    },
    "ResNet152": {
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resnet152_tv_in1k.weights.h5",
        },
        "a1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resnet152_a1_in1k.weights.h5",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resnet152_gluon_in1k.weights.h5",
        },
    },
}
