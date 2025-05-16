SEGFORMER_MODEL_CONFIG = {
    "SegFormerB0": {"embed_dim": 256, "dropout_rate": 0.1},
    "SegFormerB1": {"embed_dim": 256, "dropout_rate": 0.1},
    "SegFormerB2": {"embed_dim": 768, "dropout_rate": 0.1},
    "SegFormerB3": {"embed_dim": 768, "dropout_rate": 0.1},
    "SegFormerB4": {"embed_dim": 768, "dropout_rate": 0.1},
    "SegFormerB5": {"embed_dim": 768, "dropout_rate": 0.1},
}

SEGFORMER_WEIGHTS_CONFIG = {
    "SegFormerB0": {
        "cityscapes_1024": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/SegFormer_B0_city_1024.weights.h5",
        },
        "cityscapes_768": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/SegFormer_B0_city_768.weights.h5",
        },
        "ade20k_512": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/SegFormer_B0_ade_512.weights.h5",
        },
    },
    "SegFormerB1": {
        "cityscapes_1024": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/SegFormer_B1_city_1024.weights.h5",
        },
        "ade20k_512": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/SegFormer_B1_ade_512.weights.h5",
        },
    },
    "SegFormerB2": {
        "cityscapes_1024": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/SegFormer_B2_city_1024.weights.h5",
        },
        "ade20k_512": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/SegFormer_B2_ade_512.weights.h5",
        },
    },
    "SegFormerB3": {
        "cityscapes_1024": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/SegFormer_B3_city_1024.weights.h5",
        },
        "ade20k_512": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/SegFormer_B3_ade_512.weights.h5",
        },
    },
    "SegFormerB4": {
        "cityscapes_1024": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/SegFormer_B4_city_1024.weights.h5",
        },
        "ade20k_512": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/SegFormer_B4_ade_512.weights.h5",
        },
    },
    "SegFormerB5": {
        "cityscapes_1024": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/SegFormer_B5_city_1024.weights.h5",
        },
        "ade20k_512": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/SegFormer_B5_ade_640.weights.h5",
        },
    },
}
