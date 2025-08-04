INCEPTION_NEXT_MODEL_CONFIG = {
    "InceptionNeXtAtto": {
        "depths": [2, 2, 6, 2],
        "num_filters": [40, 80, 160, 320],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 9,
        "branch_ratio": 0.25,
    },
    "InceptionNeXtTiny": {
        "depths": [3, 3, 9, 3],
        "num_filters": [96, 192, 384, 768],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
    },
    "InceptionNeXtSmall": {
        "depths": [3, 3, 27, 3],
        "num_filters": [96, 192, 384, 768],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
    },
    "InceptionNeXtBase": {
        "depths": [3, 3, 27, 3],
        "num_filters": [128, 256, 512, 1024],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
    },
}

INCEPTION_NEXT_WEIGHTS_CONFIG = {
    "InceptionNeXtAtto": {
        "sail_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/inception_next_atto_sail_in1k.weights.h5"
        },
    },
    "InceptionNeXtTiny": {
        "sail_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/inception_next_tiny_sail_in1k.weights.h5"
        },
    },
    "InceptionNeXtSmall": {
        "sail_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/inception_next_small_sail_in1k.weights.h5"
        },
    },
    "InceptionNeXtBase": {
        "sail_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/inception_next_base_sail_in1k.weights.h5"
        },
    },
}
