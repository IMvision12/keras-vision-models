RES2NET_MODEL_CONFIG = {
    "Res2Net50_26w_4s": {
        "depth": [3, 4, 6, 3],
        "base_width": 26,
        "scale": 4,
        "cardinality": 1,
    },
    "Res2Net101_26w_4s": {
        "depth": [3, 4, 23, 3],
        "base_width": 26,
        "scale": 4,
        "cardinality": 1,
    },
    "Res2Net50_26w_6s": {
        "depth": [3, 4, 6, 3],
        "base_width": 26,
        "scale": 6,
        "cardinality": 1,
    },
    "Res2Net50_26w_8s": {
        "depth": [3, 4, 6, 3],
        "base_width": 26,
        "scale": 8,
        "cardinality": 1,
    },
    "Res2Net50_48w_2s": {
        "depth": [3, 4, 6, 3],
        "base_width": 48,
        "scale": 2,
        "cardinality": 1,
    },
    "Res2Net50_14w_8s": {
        "depth": [3, 4, 6, 3],
        "base_width": 14,
        "scale": 8,
        "cardinality": 1,
    },
    "Res2Next50": {
        "depth": [3, 4, 6, 3],
        "base_width": 4,
        "scale": 4,
        "cardinality": 8,
    },
}

RES2NET_WEIGHTS_CONFIG = {
    "Res2Net50_26w_4s": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/res2net50_26w_4s_in1k.weights.h5"
        },
    },
    "Res2Net101_26w_4s": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/res2net101_26w_4s_in1k.weights.h5"
        },
    },
    "Res2Net50_26w_6s": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/res2net50_26w_6s_in1k.weights.h5"
        },
    },
    "Res2Net50_26w_8s": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/res2net50_26w_8s_in1k.weights.h5"
        },
    },
    "Res2Net50_48w_2s": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/res2net50_48w_2s_in1k.weights.h5"
        },
    },
    "Res2Net50_14w_8s": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/res2net50_14w_8s_in1k.weights.h5"
        },
    },
    "Res2Next50": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/res2next50_in1k.weights.h5"
        },
    },
}
