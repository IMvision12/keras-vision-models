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
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/seresnet50_a1_in1k.weights.h5",
        }
    },
    "SEResNeXt50_32x4d": {
        "racm_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/seresnext50_32x4d_racm_in1k.weights.h5",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/seresnext50_32x4d_gluon_in1k.weights.h5",
        },
    },
    "SEResNeXt101_32x4d": {
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/seresnext101_32x4d_gluon_in1k.weights.h5",
        },
    },
    "SEResNeXt101_32x8d": {
        "ah_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/seresnext101_32x8d_ah_in1k.weights.h5",
        }
    },
}
