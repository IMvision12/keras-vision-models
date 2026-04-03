SWINV2_MODEL_CONFIG = {
    "SwinV2TinyW8": {
        "pretrain_size": 256,
        "window_size": 8,
        "embed_dim": 96,
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
    },
    "SwinV2TinyW16": {
        "pretrain_size": 256,
        "window_size": 16,
        "embed_dim": 96,
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
    },
    "SwinV2SmallW8": {
        "pretrain_size": 256,
        "window_size": 8,
        "embed_dim": 96,
        "depths": (2, 2, 18, 2),
        "num_heads": (3, 6, 12, 24),
    },
    "SwinV2SmallW16": {
        "pretrain_size": 256,
        "window_size": 16,
        "embed_dim": 96,
        "depths": (2, 2, 18, 2),
        "num_heads": (3, 6, 12, 24),
    },
    "SwinV2BaseW8": {
        "pretrain_size": 256,
        "window_size": 8,
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
    },
    "SwinV2BaseW12": {
        "pretrain_size": 192,
        "window_size": 12,
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
    },
    "SwinV2BaseW16": {
        "pretrain_size": 256,
        "window_size": 16,
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
    },
    "SwinV2LargeW12": {
        "pretrain_size": 192,
        "window_size": 12,
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
    },
}

SWINV2_WEIGHTS_CONFIG = {
    "SwinV2TinyW8": {
        "ms_in1k": {
            "url": "",
        },
    },
    "SwinV2TinyW16": {
        "ms_in1k": {
            "url": "",
        },
    },
    "SwinV2SmallW8": {
        "ms_in1k": {
            "url": "",
        },
    },
    "SwinV2SmallW16": {
        "ms_in1k": {
            "url": "",
        },
    },
    "SwinV2BaseW8": {
        "ms_in1k": {
            "url": "",
        },
    },
    "SwinV2BaseW12": {
        "ms_in22k": {
            "url": "",
        },
        "ms_in22k_ft_in1k_256": {
            "url": "",
            "model_kwargs": {"window_size": 16, "pretrained_window_size": 12},
        },
        "ms_in22k_ft_in1k_384": {
            "url": "",
            "model_kwargs": {"window_size": 24, "pretrained_window_size": 12},
        },
    },
    "SwinV2BaseW16": {
        "ms_in1k": {
            "url": "",
        },
    },
    "SwinV2LargeW12": {
        "ms_in22k": {
            "url": "",
        },
        "ms_in22k_ft_in1k_256": {
            "url": "",
            "model_kwargs": {"window_size": 16, "pretrained_window_size": 12},
        },
        "ms_in22k_ft_in1k_384": {
            "url": "",
            "model_kwargs": {"window_size": 24, "pretrained_window_size": 12},
        },
    },
}
