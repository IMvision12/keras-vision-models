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
            "url": "https://github.com/IMvision12/keras-models/releases/download/nextvit/nextvit_small_bd_in1k.weights.h5",
        },
        "bd_in1k_384": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/nextvit/nextvit_small_bd_in1k_384.weights.h5",
        },
        "bd_ssld_6m_in1k_224": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/nextvit/nextvit_small_bd_ssld_6m_in1k.weights.h5",
        },
        "bd_ssld_6m_in1k_384": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/nextvit/nextvit_small_bd_ssld_6m_in1k_384.weights.h5",
        },
    },
    "NextViTBase": {
        "bd_in1k_224": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/nextvit/nextvit_base_bd_in1k.weights.h5",
        },
        "bd_in1k_384": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/nextvit/nextvit_base_bd_in1k_384.weights.h5",
        },
        "bd_ssld_6m_in1k_224": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/nextvit/nextvit_base_bd_ssld_6m_in1k.weights.h5",
        },
        "bd_ssld_6m_in1k_384": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/nextvit/nextvit_base_bd_ssld_6m_in1k_384.weights.h5",
        },
    },
    "NextViTLarge": {
        "bd_in1k_224": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/nextvit/nextvit_large_bd_in1k.weights.h5",
        },
        "bd_in1k_384": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/nextvit/nextvit_large_bd_in1k_384.weights.h5",
        },
        "bd_ssld_6m_in1k_224": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/nextvit/nextvit_large_bd_ssld_6m_in1k.weights.h5",
        },
        "bd_ssld_6m_in1k_384": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/nextvit/nextvit_large_bd_ssld_6m_in1k_384.weights.h5",
        },
    },
}
