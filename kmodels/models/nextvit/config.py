NEXTVIT_MODEL_CONFIG = {
    "NextViTSmall": {
        "depths": [3, 4, 10, 3],
        "stem_chs": [64, 32, 64],
        "head_dim": 32,
        "mix_block_ratio": 0.75,
        "sr_ratios": [8, 4, 2, 1],
        "drop_path_rate": 0.1,
    },
    "NextViTBase": {
        "depths": [3, 4, 20, 3],
        "stem_chs": [64, 32, 64],
        "head_dim": 32,
        "mix_block_ratio": 0.75,
        "sr_ratios": [8, 4, 2, 1],
        "drop_path_rate": 0.1,
    },
    "NextViTLarge": {
        "depths": [3, 4, 30, 3],
        "stem_chs": [64, 32, 64],
        "head_dim": 32,
        "mix_block_ratio": 0.75,
        "sr_ratios": [8, 4, 2, 1],
        "drop_path_rate": 0.1,
    },
}

NEXTVIT_WEIGHTS_CONFIG = {
    "NextViTSmall": {
        "bd_in1k_224": {
            "url": "",
        },
        "bd_in1k_384": {
            "url": "",
        },
        "bd_ssld_6m_in1k_224": {
            "url": "",
        },
        "bd_ssld_6m_in1k_384": {
            "url": "",
        },
    },
    "NextViTBase": {
        "bd_in1k_224": {
            "url": "",
        },
        "bd_in1k_384": {
            "url": "",
        },
        "bd_ssld_6m_in1k_224": {
            "url": "",
        },
        "bd_ssld_6m_in1k_384": {
            "url": "",
        },
    },
    "NextViTLarge": {
        "bd_in1k_224": {
            "url": "",
        },
        "bd_in1k_384": {
            "url": "",
        },
        "bd_ssld_6m_in1k_224": {
            "url": "",
        },
        "bd_ssld_6m_in1k_384": {
            "url": "",
        },
    },
}
