DENSENET_MODEL_CONFIG = {
    "DenseNet121": {
        "num_blocks": [6, 12, 24, 16],
        "growth_rate": 32,
    },
    "DenseNet161": {
        "num_blocks": [6, 12, 36, 24],
        "growth_rate": 48,
    },
    "DenseNet169": {
        "num_blocks": [6, 12, 32, 32],
        "growth_rate": 32,
    },
    "DenseNet201": {
        "num_blocks": [6, 12, 48, 32],
        "growth_rate": 32,
    },
}

DENSENET_WEIGHTS_CONFIG = {
    # DenseNet Variants
    "DenseNet121": {
        "tv_in1k": {
            "url": "",
        },
    },
    "DenseNet161": {
        "tv_in1k": {
            "url": "",
        },
    },
    "DenseNet169": {
        "tv_in1k": {
            "url": "",
        },
    },
    "DenseNet201": {
        "tv_in1k": {
            "url": "",
        },
    },
}
