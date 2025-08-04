MLPMIXER_MODEL_CONFIG = {
    "MLPMixerB16": {
        "patch_size": 16,
        "num_blocks": 12,
        "embed_dim": 768,
        "mlp_ratio": (0.5, 4.0),
    },
    "MLPMixerL16": {
        "patch_size": 16,
        "num_blocks": 24,
        "embed_dim": 1024,
        "mlp_ratio": (0.5, 4.0),
    },
}

MLPMIXER_WEIGHTS_CONFIG = {
    "MLPMixerB16": {
        "goog_in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/mixer_b16_224_goog_in21k_ft_in1k.weights.h5",
        },
        "goog_in21k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/mixer_b16_224_goog_in21k.weights.h5",
        },
        "miil_in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/mixer_b16_224_miil_in21k_ft_in1k.weights.h5",
        },
    },
    "MLPMixerL16": {
        "goog_in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/mixer_l16_224_goog_in21k_ft_in1k.weights.h5",
        },
        "goog_in21k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/mixer_l16_224_goog_in21k.weights.h5",
        },
    },
}
