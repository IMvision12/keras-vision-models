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
            "url": "",
        },
        "goog_in21k": {
            "url": "",
        },
        "miil_in21k_ft_in1k": {
            "url": "",
        },
    },
    "MLPMixerL16": {
        "goog_in21k_ft_in1k": {
            "url": "",
        },
        "goog_in21k": {
            "url": "",
        },
    },
}
