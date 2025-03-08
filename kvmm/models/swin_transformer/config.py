SWIN_MODEL_CONFIG = {
    "SwinTinyP4W7": {
        "pretrain_size": 224,
        "window_size": 7,
        "embed_dim": 96,
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
    },
    "SwinSmallP4W7": {
        "pretrain_size": 224,
        "window_size": 7,
        "embed_dim": 96,
        "depths": (2, 2, 18, 2),
        "num_heads": (3, 6, 12, 24),
    },
    "SwinBaseP4W7": {
        "pretrain_size": 224,
        "window_size": 7,
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
    },
    "SwinBaseP4W12": {
        "pretrain_size": 384,
        "window_size": 12,
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
    },
    "SwinLargeP4W7": {
        "pretrain_size": 224,
        "window_size": 7,
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
    },
    "SwinLargeP4W12": {
        "pretrain_size": 384,
        "window_size": 12,
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
    },
}

SWIN_WEIGHTS_CONFIG = {
    "SwinTinyP4W7": {
        "ms_in1k": {
            "url": "",
        },
        "ms_in22k": {
            "url": "",
        },
    },
    "SwinSmallP4W7": {
        "ms_in22k_ft_in1k": {
            "url": "",
        },
        "ms_in1k": {
            "url": "",
        },
        "ms_in22k": {
            "url": "",
        },
    },
    "SwinBaseP4W7": {
        "ms_in22k_ft_in1k": {
            "url": "",
        },
        "ms_in1k": {
            "url": "",
        },
        "ms_in22k": {
            "url": "",
        },
    },
    "SwinBaseP4W12": {
        "ms_in22k_ft_in1k": {
            "url": "",
        },
        "ms_in1k": {
            "url": "",
        },
        "ms_in22k": {
            "url": "",
        },
    },
    "SwinLargeP4W7": {
        "ms_in22k_ft_in1k": {
            "url": "",
        },
        "ms_in22k": {
            "url": "",
        },
    },
    "SwinLargeP4W12": {
        "ms_in22k_ft_in1k": {
            "url": "",
        },
        "ms_in22k": {
            "url": "",
        },
    },
}
