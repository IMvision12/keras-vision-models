CLIP_MODEL_CONFIG = {
    "ClipVitBase16": {
        "embed_dim": 512,
        "vision_layers": 12,
        "vision_width": 768,
        "vision_patch_size": 16,
        "context_length": 77,
        "vocab_size": 49408,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
    },
    "ClipVitBase32": {
        "embed_dim": 512,
        "vision_layers": 12,
        "vision_width": 768,
        "vision_patch_size": 32,
        "context_length": 77,
        "vocab_size": 49408,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
    },
    "ClipVitLarge14": {
        "embed_dim": 768,
        "vision_layers": 24,
        "vision_width": 1024,
        "vision_patch_size": 14,
        "context_length": 77,
        "vocab_size": 49408,
        "transformer_width": 768,
        "transformer_heads": 12,
        "transformer_layers": 12,
    },
    "ClipVitG14": {
        "embed_dim": 1024,
        "vision_layers": 40,
        "vision_width": 1408,
        "vision_patch_size": 14,
        "context_length": 77,
        "vocab_size": 49408,
        "transformer_width": 1024,
        "transformer_heads": 16,
        "transformer_layers": 24,
    },
    "ClipVitBigG14": {
        "embed_dim": 1280,
        "vision_layers": 48,
        "vision_width": 1664,
        "vision_patch_size": 14,
        "context_length": 77,
        "vocab_size": 49408,
        "transformer_width": 1280,
        "transformer_heads": 20,
        "transformer_layers": 32,
    },
}


CLIP_WEIGHTS_CONFIG = {
    "ClipVitBase16": {
        "openai_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/clip/clipvitbase16_224.weights.h5",
        },
    },
    "ClipVitBase32": {
        "openai_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/clip/clipvitbase32_224.weights.h5",
        },
        "laion2b_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/clip/clipvitlarge14_224.weights.h5",
        },
    },
    "ClipVitLarge14": {
        "openai_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/clip/clipvitlarge14_224.weights.h5",
        },
        "openai_336": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/clip/clipvitlarge14_336.weights.h5",
        },
        "laion2b_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/clip/clipvitlarge14_224.weights.h5",
        },
    },
    "ClipVitG14": {
        "laion2b_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/clip/clipvitlarge14_224.weights.h5",
        },
    },
    "ClipVitBigG14": {
        "laion2b_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/clip/clipvitlarge14_224.weights.h5",
        },
    },
}
