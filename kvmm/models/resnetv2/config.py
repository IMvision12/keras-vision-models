RESNETV2_MODEL_CONFIG = {
    "ResNetV2_50x1": {
        "block_repeats": [3, 4, 6, 3],
        "width_factor": 1,
    },
    "ResNetV2_50x3": {
        "block_repeats": [3, 4, 6, 3],
        "width_factor": 3,
    },
    "ResNetV2_101x1": {
        "block_repeats": [3, 4, 23, 3],
        "width_factor": 1,
    },
    "ResNetV2_101x3": {
        "block_repeats": [3, 4, 23, 3],
        "width_factor": 3,
    },
    "ResNetV2_152x2": {
        "block_repeats": [3, 8, 36, 3],
        "width_factor": 2,
    },
    "ResNetV2_152x4": {
        "block_repeats": [3, 8, 36, 3],
        "width_factor": 4,
    },
}

RESNETV2_WEIGHTS_CONFIG = {
    "ResNetV2_50x1": {
        "goog_in21k_ft_in1k_448": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/resnetv2_50x1_bit_goog_in21k_ft_in1k.weights.h5",
        },
        "goog_in21k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/resnetv2_50x1_bit_goog_in21k.weights.h5",
        },
    },
    "ResNetV2_50x3": {
        "goog_in21k_ft_in1k_448": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/resnetv2_50x3_bit_goog_in21k_ft_in1k.weights.h5",
        },
        "goog_in21k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/resnetv2_50x3_bit_goog_in21k.weights.h5",
        },
    },
    "ResNetV2_101x1": {
        "goog_in21k_ft_in1k_448": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/resnetv2_101x1_bit_goog_in21k_ft_in1k.weights.h5",
        },
        "goog_in21k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/resnetv2_101x1_bit_goog_in21k.weights.h5",
        },
    },
    "ResNetV2_101x3": {
        "goog_in21k_ft_in1k_448": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/resnetv2_101x3_bit_goog_in21k_ft_in1k.weights.h5",
        },
        "goog_in21k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/resnetv2_101x3_bit_goog_in21k.weights.h5",
        },
    },
    "ResNetV2_152x2": {
        "goog_in21k_ft_in1k_448": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/resnetv2_152x2_bit_goog_in21k_ft_in1k.weights.h5",
        },
        "goog_in21k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/resnetv2_152x2_bit_goog_in21k.weights.h5",
        },
    },
    "ResNetV2_152x4": {
        "goog_in21k_ft_in1k_480": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/resnetv2_152x4_bit_goog_in21k_ft_in1k.weights.json",
        },
        "goog_in21k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/resnetv2_152x4_bit_goog_in21k.weights.json",
        },
    },
}
