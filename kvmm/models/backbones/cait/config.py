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
        "fb_dist_in1k_224": {"url": ""},
        "fb_dist_in1k_384": {"url": ""},
    },
    "CaiTXXS36": {
        "fb_dist_in1k_224": {"url": ""},
        "fb_dist_in1k_384": {"url": ""},
    },
    "CaiTXS24": {
        "fb_dist_in1k_384": {"url": ""},
    },
    "CaiTS24": {
        "fb_dist_in1k_224": {"url": ""},
        "fb_dist_in1k_384": {"url": ""},
    },
    "CaiTS36": {
        "fb_dist_in1k_384": {"url": ""},
    },
    "CaiTM36": {
        "fb_dist_in1k_384": {"url": ""},
    },
    "CaiTM48": {
        "fb_dist_in1k_448": {"url": ""},
    },
}
