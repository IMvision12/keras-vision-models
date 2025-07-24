RESMLP_MODEL_CONFIG = {
    "ResMLP12": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "mlp_ratio": 4,
        "init_values": 1e-4,
    },
    "ResMLP24": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 24,
        "mlp_ratio": 4,
        "init_values": 1e-5,
    },
    "ResMLP36": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 36,
        "mlp_ratio": 4,
        "init_values": 1e-6,
    },
    "ResMLPBig24": {
        "patch_size": 8,
        "embed_dim": 768,
        "depth": 24,
        "mlp_ratio": 4,
        "init_values": 1e-6,
    },
}

RESMLP_WEIGHTS_CONFIG = {
    "ResMLP12": {
        "fb_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resmlp_12_224_fb_in1k.weights.h5"
        },
        "fb_dist_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resmlp_12_224_fb_distilled_in1k.weights.h5"
        },
    },
    "ResMLP24": {
        "fb_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resmlp_24_224_fb_in1k.weights.h5"
        },
        "fb_dist_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resmlp_24_224_fb_distilled_in1k.weights.h5"
        },
    },
    "ResMLP36": {
        "fb_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resmlp_36_224_fb_in1k.weights.h5"
        },
        "fb_dist_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resmlp_36_224_fb_distilled_in1k.weights.h5"
        },
    },
    "ResMLPBig24": {
        "fb_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resmlp_big_24_224_fb_in1k.weights.h5"
        },
        "fb_dist_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resmlp_big_24_224_fb_distilled_in1k.weights.h5"
        },
        "fb_in22k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/resmlp_big_24_224_fb_in22k_ft_in1k.weights.h5"
        },
    },
}
