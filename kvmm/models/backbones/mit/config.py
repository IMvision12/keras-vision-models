MIT_MODEL_CONFIG = {
    "MiT_B0": {"embed_dims": [32, 64, 160, 256], "depths": [2, 2, 2, 2]},
    "MiT_B1": {"embed_dims": [64, 128, 320, 512], "depths": [2, 2, 2, 2]},
    "MiT_B2": {"embed_dims": [64, 128, 320, 512], "depths": [3, 4, 6, 3]},
    "MiT_B3": {"embed_dims": [64, 128, 320, 512], "depths": [3, 4, 18, 3]},
    "MiT_B4": {"embed_dims": [64, 128, 320, 512], "depths": [3, 8, 27, 3]},
    "MiT_B5": {"embed_dims": [64, 128, 320, 512], "depths": [3, 6, 40, 3]},
}

MIT_WEIGHTS_CONFIG = {
    "MiT_B0": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/MiT_B0.weights.h5"
        },
    },
    "MiT_B1": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/MiT_B1.weights.h5"
        },
    },
    "MiT_B2": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/MiT_B2.weights.h5"
        },
    },
    "MiT_B3": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/MiT_B3.weights.h5"
        },
    },
    "MiT_B4": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/MiT_B4.weights.h5"
        },
    },
    "MiT_B5": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.3/MiT_B5.weights.h5"
        },
    },
}
