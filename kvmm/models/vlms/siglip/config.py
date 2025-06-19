SigLIP_MODEL_CONFIG = {
    "SigLIPBaseP16": {
        "patch_size": 16,
        "vision_hidden_dim": 768,
        "vision_num_layers": 12,
        "vision_num_heads": 12,
        "vision_intermediate_dim": 3072,
        "vocabulary_size": 32000,
        "embed_dim": 768,
        "text_hidden_dim": 768,
        "text_num_layers": 12,
        "text_num_heads": 12,
        "text_intermediate_dim": 3072,
    },
    "SigLIPLargeP16": {
        "patch_size": 16,
        "vision_hidden_dim": 1024,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "vision_intermediate_dim": 4096,
        "vocabulary_size": 32000,
        "embed_dim": 1024,
        "text_hidden_dim": 1024,
        "text_num_layers": 24,
        "text_num_heads": 16,
        "text_intermediate_dim": 4096,
    },
    "SigLIPSo400mP14": {
        "patch_size": 14,
        "vision_hidden_dim": 1152,
        "vision_num_layers": 27,
        "vision_num_heads": 16,
        "vision_intermediate_dim": 4304,
        "vocabulary_size": 32000,
        "embed_dim": 1152,
        "text_hidden_dim": 1152,
        "text_num_layers": 27,
        "text_num_heads": 16,
        "text_intermediate_dim": 4304,
    },
}


SigLIP_WEIGHTS_CONFIG = {
    "SigLIPBaseP16": {
        "google_224": {
            "url": "",
        },
        "google_256": {
            "url": "",
        },
        "google_384": {
            "url": "",
        },
        "google_512": {
            "url": "",
        },
    },
    "SigLIPLargeP16": {
        "google_256": {
            "url": "",
        },
        "google_384": {
            "url": "",
        },
    },
    "SigLIPSo400mP14": {
        "google_224": {
            "url": "",
        },
        "google_384": {
            "url": "",
        },
    },
}
