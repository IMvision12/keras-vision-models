PIT_MODEL_CONFIG = {
    "PiT_XS": {
        "patch_size": 16,
        "stride": 8,
        "embed_dim": [96, 192, 384],
        "depth": [2, 6, 4],
        "heads": [2, 4, 8],
        "mlp_ratio": 4,
        "distilled": False,
    },
    "PiT_XS_Distilled": {
        "patch_size": 16,
        "stride": 8,
        "embed_dim": [96, 192, 384],
        "depth": [2, 6, 4],
        "heads": [2, 4, 8],
        "mlp_ratio": 4,
        "distilled": True,
    },
    "PiT_Ti": {
        "patch_size": 16,
        "stride": 8,
        "embed_dim": [64, 128, 256],
        "depth": [2, 6, 4],
        "heads": [2, 4, 8],
        "mlp_ratio": 4,
        "distilled": False,
    },
    "PiT_Ti_Distilled": {
        "patch_size": 16,
        "stride": 8,
        "embed_dim": [64, 128, 256],
        "depth": [2, 6, 4],
        "heads": [2, 4, 8],
        "mlp_ratio": 4,
        "distilled": True,
    },
    "PiT_S": {
        "patch_size": 16,
        "stride": 8,
        "embed_dim": [144, 288, 576],
        "depth": [2, 6, 4],
        "heads": [3, 6, 12],
        "mlp_ratio": 4,
        "distilled": False,
    },
    "PiT_S_Distilled": {
        "patch_size": 16,
        "stride": 8,
        "embed_dim": [144, 288, 576],
        "depth": [2, 6, 4],
        "heads": [3, 6, 12],
        "mlp_ratio": 4,
        "distilled": True,
    },
    "PiT_B": {
        "patch_size": 14,
        "stride": 7,
        "embed_dim": [256, 512, 1024],
        "depth": [3, 6, 4],
        "heads": [4, 8, 16],
        "mlp_ratio": 4,
        "distilled": False,
    },
    "PiT_B_Distilled": {
        "patch_size": 14,
        "stride": 7,
        "embed_dim": [256, 512, 1024],
        "depth": [3, 6, 4],
        "heads": [4, 8, 16],
        "mlp_ratio": 4,
        "distilled": True,
    },
}


PIT_WEIGHTS_CONFIG = {
    "PiT_XS": {
        "in1k": {"url": ""},
    },
    "PiT_XS_Distilled": {
        "in1k": {"url": ""},
    },
    "PiT_Ti": {
        "in1k": {"url": ""},
    },
    "PiT_Ti_Distilled": {
        "in1k": {"url": ""},
    },
    "PiT_S": {
        "in1k": {"url": ""},
    },
    "PiT_S_Distilled": {
        "in1k": {"url": ""},
    },
    "PiT_B": {
        "in1k": {"url": ""},
    },
    "PiT_B_Distilled": {
        "in1k": {"url": ""},
    },
}
