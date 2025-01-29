INCEPTION_NEXT_MODEL_CONFIG = {
    "InceptionNeXtAtto": {
        "depths": [2, 2, 6, 2],
        "num_filters": [40, 80, 160, 320],
        "mlp_ratios": [4, 4, 4, 3],
    },
    "InceptionNeXtTiny": {
        "depths": [3, 3, 9, 3],
        "num_filters": [96, 192, 384, 768],
        "mlp_ratios": [4, 4, 4, 3],
    },
    "InceptionNeXtSmall": {
        "depths": [3, 3, 27, 3],
        "num_filters": [96, 192, 384, 768],
        "mlp_ratios": [4, 4, 4, 3],
    },
    "InceptionNeXtBase": {
        "depths": [3, 3, 27, 3],
        "num_filters": [128, 256, 512, 1024],
        "mlp_ratios": [4, 4, 4, 3],
    },
}

INCEPTION_NEXT_WEIGHTS_CONFIG = {
    "InceptionNeXtAtto": {
        "sail_in1k": {"url": ""},
    },
    "InceptionNeXtTiny": {
        "sail_in1k": {"url": ""},
    },
    "InceptionNeXtSmall": {
        "sail_in1k": {"url": ""},
    },
    "InceptionNeXtBase": {
        "sail_in1k": {"url": ""},
    },
}
