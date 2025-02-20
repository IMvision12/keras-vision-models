MOBILEVIT_MODEL_CONFIG = {
    "MobileViTXXS": {
        "initial_dims": 16,
        "head_dims": 320,
        "block_dims": [16, 24, 48, 64, 80],
        "expansion_ratio": [2.0, 2.0, 2.0, 2.0, 2.0],
        "attention_dims": [None, None, 64, 80, 96],
    },
    "MobileViTXS": {
        "initial_dims": 16,
        "head_dims": 384,
        "block_dims": [32, 48, 64, 80, 96],
        "expansion_ratio": [4.0, 4.0, 4.0, 4.0, 4.0],
        "attention_dims": [None, None, 96, 120, 144],
    },
    "MobileViTS": {
        "initial_dims": 16,
        "head_dims": 640,
        "block_dims": [32, 64, 96, 128, 160],
        "expansion_ratio": [4.0, 4.0, 4.0, 4.0, 4.0],
        "attention_dims": [None, None, 144, 192, 240],
    }
}

MOBILEVIT_WEIGHTS_CONFIG = {
    "MobileViTXXS": {
        "cvnets_in1k": {"url": ""},
    },
    "MobileViTXS": {
        "cvnets_in1k": {"url": ""},
    },
    "MobileViTS": {
        "cvnets_in1k": {"url": ""},
    },
}
