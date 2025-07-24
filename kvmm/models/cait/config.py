CAIT_MODEL_CONFIG = {
    "CaiTXXS24": {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 24,
        "num_heads": 4,
        "init_values": 1e-5,
    },
    "CaiTXXS36": {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 36,
        "num_heads": 4,
        "init_values": 1e-5,
    },
    "CaiTXS24": {
        "patch_size": 16,
        "embed_dim": 288,
        "depth": 24,
        "num_heads": 6,
        "init_values": 1e-5,
    },
    "CaiTS24": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 24,
        "num_heads": 8,
        "init_values": 1e-5,
    },
    "CaiTS36": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 36,
        "num_heads": 8,
        "init_values": 1e-6,
    },
    "CaiTM36": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 36,
        "num_heads": 16,
        "init_values": 1e-6,
    },
    "CaiTM48": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 48,
        "num_heads": 16,
        "init_values": 1e-6,
    },
}

CAIT_WEIGHTS_CONFIG = {
    "CaiTXXS24": {
        "fb_dist_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/cait_xxs24_224_fb_dist_in1k.weights.h5"
        },
        "fb_dist_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/cait_xxs24_384_fb_dist_in1k.weights.h5"
        },
    },
    "CaiTXXS36": {
        "fb_dist_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/cait_xxs36_224_fb_dist_in1k.weights.h5"
        },
        "fb_dist_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/cait_xxs36_384_fb_dist_in1k.weights.h5"
        },
    },
    "CaiTXS24": {
        "fb_dist_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/cait_xs24_384_fb_dist_in1k.weights.h5"
        },
    },
    "CaiTS24": {
        "fb_dist_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/cait_s24_224_fb_dist_in1k.weights.h5"
        },
        "fb_dist_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/cait_s24_384_fb_dist_in1k.weights.h5"
        },
    },
    "CaiTS36": {
        "fb_dist_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/cait_s36_384_fb_dist_in1k.weights.h5"
        },
    },
    "CaiTM36": {
        "fb_dist_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/cait_m36_384_fb_dist_in1k.weights.h5"
        },
    },
    "CaiTM48": {
        "fb_dist_in1k_448": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/cait_m48_448_fb_dist_in1k.weights.h5"
        },
    },
}
