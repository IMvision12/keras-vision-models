SAM_MODEL_CONFIG = {
    "SAMViTBase": {
        "vision_hidden_size": 768,
        "vision_num_hidden_layers": 12,
        "vision_num_attention_heads": 12,
        "vision_mlp_dim": 3072,
        "vision_global_attn_indexes": [2, 5, 8, 11],
    },
    "SAMViTLarge": {
        "vision_hidden_size": 1024,
        "vision_num_hidden_layers": 24,
        "vision_num_attention_heads": 16,
        "vision_mlp_dim": 4096,
        "vision_global_attn_indexes": [5, 11, 17, 23],
    },
    "SAMViTHuge": {
        "vision_hidden_size": 1280,
        "vision_num_hidden_layers": 32,
        "vision_num_attention_heads": 16,
        "vision_mlp_dim": 5120,
        "vision_global_attn_indexes": [7, 15, 23, 31],
    },
}

SAM_WEIGHTS_CONFIG = {
    "SAMViTBase": {
        "sa1b": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/sam/sam_vit_base.weights.h5",
        },
    },
    "SAMViTLarge": {
        "sa1b": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/sam/sam_vit_large.weights.h5",
        },
    },
    "SAMViTHuge": {
        "sa1b": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/sam/sam_vit_huge.weights.json",
        },
    },
}
