SigLIP2_MODEL_CONFIG = {
    "SigLIP2BaseP16": {
        "patch_size": 16,
        "vision_hidden_dim": 768,
        "vision_num_layers": 12,
        "vision_num_heads": 12,
        "vision_intermediate_dim": 3072,
        "vocabulary_size": 256000,
        "embed_dim": 768,
        "text_hidden_dim": 768,
        "text_num_layers": 12,
        "text_num_heads": 12,
        "text_intermediate_dim": 3072,
        "max_sequence_length": 64,
    },
    "SigLIP2BaseP32": {
        "patch_size": 32,
        "vision_hidden_dim": 768,
        "vision_num_layers": 12,
        "vision_num_heads": 12,
        "vision_intermediate_dim": 3072,
        "vocabulary_size": 256000,
        "embed_dim": 768,
        "text_hidden_dim": 768,
        "text_num_layers": 12,
        "text_num_heads": 12,
        "text_intermediate_dim": 3072,
        "max_sequence_length": 64,
    },
    "SigLIP2LargeP16": {
        "patch_size": 16,
        "vision_hidden_dim": 1024,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "vision_intermediate_dim": 4096,
        "vocabulary_size": 256000,
        "embed_dim": 1024,
        "text_hidden_dim": 1024,
        "text_num_layers": 24,
        "text_num_heads": 16,
        "text_intermediate_dim": 4096,
        "max_sequence_length": 64,
    },
    "SigLIP2So400mP14": {
        "patch_size": 14,
        "vision_hidden_dim": 1152,
        "vision_num_layers": 27,
        "vision_num_heads": 16,
        "vision_intermediate_dim": 4304,
        "vocabulary_size": 256000,
        "embed_dim": 1152,
        "text_hidden_dim": 1152,
        "text_num_layers": 27,
        "text_num_heads": 16,
        "text_intermediate_dim": 4304,
        "max_sequence_length": 64,
    },
    "SigLIP2So400mP16": {
        "patch_size": 16,
        "vision_hidden_dim": 1152,
        "vision_num_layers": 27,
        "vision_num_heads": 16,
        "vision_intermediate_dim": 4304,
        "vocabulary_size": 256000,
        "embed_dim": 1152,
        "text_hidden_dim": 1152,
        "text_num_layers": 27,
        "text_num_heads": 16,
        "text_intermediate_dim": 4304,
        "max_sequence_length": 64,
    },
}


SigLIP2_WEIGHTS_CONFIG = {
    "SigLIP2BaseP16": {
        "google_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2basep16_google_224.weights.h5",
        },
        "google_256": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2basep16_google_256.weights.h5",
        },
        "google_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2basep16_google_384.weights.h5",
        },
        "google_512": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2basep16_google_512.weights.h5",
        },
    },
    "SigLIP2BaseP32": {
        "google_256": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2basep32_google_256.weights.h5",
        },
    },
    "SigLIP2LargeP16": {
        "google_256": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2largep16_google_256.weights.json",
        },
        "google_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2largep16_google_384.weights.json",
        },
        "google_512": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2largep16_google_512.weights.json",
        },
    },
    "SigLIP2So400mP14": {
        "google_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2so400mp14_google_224.weights.json",
        },
        "google_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2so400mp14_google_384.weights.json",
        },
    },
    "SigLIP2So400mP16": {
        "google_256": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2so400mp16_google_256.weights.json",
        },
        "google_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2so400mp16_google_384.weights.json",
        },
        "google_512": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2so400mp16_google_512.weights.json",
        },
    },
}
