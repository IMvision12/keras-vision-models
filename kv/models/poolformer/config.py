POOLFORMER_MODEL_CONFIG = {
    "PoolFormerS12": {
        "embed_dims": (64, 128, 320, 512),
        "num_blocks": (2, 2, 6, 2),
        "init_scale": 1e-5,
    },
    "PoolFormerS24": {
        "embed_dims": (64, 128, 320, 512),
        "num_blocks": (4, 4, 12, 4),
        "init_scale": 1e-5,
    },
    "PoolFormerS36": {
        "embed_dims": (64, 128, 320, 512),
        "num_blocks": (6, 6, 18, 6),
        "init_scale": 1e-6,
    },
    "PoolFormerM36": {
        "embed_dims": (96, 192, 384, 768),
        "num_blocks": (6, 6, 18, 6),
        "init_scale": 1e-6,
    },
    "PoolFormerM48": {
        "embed_dims": (96, 192, 384, 768),
        "num_blocks": (8, 8, 24, 8),
        "init_scale": 1e-6,
    },
}

POOLFORMER_WEIGHTS_CONFIG = {
    "PoolFormerS12": {
        "sail_in1k": {"url": ""},
    },
    "PoolFormerS24": {
        "sail_in1k": {"url": ""},
    },
    "PoolFormerS36": {
        "sail_in1k": {"url": ""},
    },
    "PoolFormerM36": {
        "sail_in1k": {"url": ""},
    },
    "PoolFormerM48": {
        "sail_in1k": {"url": ""},
    },
}
