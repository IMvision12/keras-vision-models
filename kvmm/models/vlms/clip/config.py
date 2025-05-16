CLIP_MODEL_CONFIG = {
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
}


CLIP_WEIGHTS_CONFIG = {
    "ClipVitBase32": {
        "res_224px": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/clip/clipvitbase32_224.weights.h5",
        },
    },
    "ClipVitBase16": {
        "res_224px": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/clip/clipvitbase16_224.weights.h5",
        },
    },
    "ClipVitLarge14": {
        "res_224px": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/clip/clipvitlarge14_224.weights.h5",
        },
        "res_336px": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/clip/clipvitlarge14_336.weights.h5",
        },
    },
}
