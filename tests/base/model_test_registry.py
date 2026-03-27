MODEL_TEST_CONFIGS = {
    "CaiTXXS24": {
        "module": "kmodels.models.cait",
        "model_cls": "CaiTXXS24",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "ConvMixer768D32": {
        "module": "kmodels.models.convmixer",
        "model_cls": "ConvMixer768D32",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "ConvNeXtAtto": {
        "module": "kmodels.models.convnext",
        "model_cls": "ConvNeXtAtto",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "ConvNeXtV2Atto": {
        "module": "kmodels.models.convnextv2",
        "model_cls": "ConvNeXtV2Atto",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "DEiTTiny16": {
        "module": "kmodels.models.deit",
        "model_cls": "DEiTTiny16",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "DenseNet121": {
        "module": "kmodels.models.densenet",
        "model_cls": "DenseNet121",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "EfficientNetB0": {
        "module": "kmodels.models.efficientnet",
        "model_cls": "EfficientNetB0",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "EfficientNetLite0": {
        "module": "kmodels.models.efficientnet_lite",
        "model_cls": "EfficientNetLite0",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "EfficientNetV2B0": {
        "module": "kmodels.models.efficientnetv2",
        "model_cls": "EfficientNetV2B0",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "EfficientFormerL1": {
        "module": "kmodels.models.efficientformer",
        "model_cls": "EfficientFormerL1",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "FlexiViTSmall": {
        "module": "kmodels.models.flexivit",
        "model_cls": "FlexiViTSmall",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "InceptionNeXtAtto": {
        "module": "kmodels.models.inception_next",
        "model_cls": "InceptionNeXtAtto",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "InceptionResNetV2": {
        "module": "kmodels.models.inception_resnetv2",
        "model_cls": "InceptionResNetV2",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (75, 75, 3),
            "include_top": True,
        },
        "input_shape": (2, 75, 75, 3),
        "expected_output_shape": (2, 1000),
    },
    "InceptionV3": {
        "module": "kmodels.models.inceptionv3",
        "model_cls": "InceptionV3",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (75, 75, 3),
            "include_top": True,
        },
        "input_shape": (2, 75, 75, 3),
        "expected_output_shape": (2, 1000),
    },
    "InceptionV4": {
        "module": "kmodels.models.inceptionv4",
        "model_cls": "InceptionV4",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (75, 75, 3),
            "include_top": True,
        },
        "input_shape": (2, 75, 75, 3),
        "expected_output_shape": (2, 1000),
    },
    "MiT_B0": {
        "module": "kmodels.models.mit",
        "model_cls": "MiT_B0",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "MLPMixerB16": {
        "module": "kmodels.models.mlp_mixer",
        "model_cls": "MLPMixerB16",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "MobileNetV2WM50": {
        "module": "kmodels.models.mobilenetv2",
        "model_cls": "MobileNetV2WM50",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "MobileNetV3Small075": {
        "module": "kmodels.models.mobilenetv3",
        "model_cls": "MobileNetV3Small075",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "MobileViTXXS": {
        "module": "kmodels.models.mobilevit",
        "model_cls": "MobileViTXXS",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "MobileViTV2M050": {
        "module": "kmodels.models.mobilevitv2",
        "model_cls": "MobileViTV2M050",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "PiT_Ti": {
        "module": "kmodels.models.pit",
        "model_cls": "PiT_Ti",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "PoolFormerS12": {
        "module": "kmodels.models.poolformer",
        "model_cls": "PoolFormerS12",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "Res2Net50_14w_8s": {
        "module": "kmodels.models.res2net",
        "model_cls": "Res2Net50_14w_8s",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "ResMLP12": {
        "module": "kmodels.models.resmlp",
        "model_cls": "ResMLP12",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "ResNet50": {
        "module": "kmodels.models.resnet",
        "model_cls": "ResNet50",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "ResNetV2_50x1": {
        "module": "kmodels.models.resnetv2",
        "model_cls": "ResNetV2_50x1",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "ResNeXt50_32x4d": {
        "module": "kmodels.models.resnext",
        "model_cls": "ResNeXt50_32x4d",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "SEResNet50": {
        "module": "kmodels.models.senet",
        "model_cls": "SEResNet50",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "SwinTinyP4W7": {
        "module": "kmodels.models.swin",
        "model_cls": "SwinTinyP4W7",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "VGG16": {
        "module": "kmodels.models.vgg",
        "model_cls": "VGG16",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "ViTTiny16": {
        "module": "kmodels.models.vit",
        "model_cls": "ViTTiny16",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "include_top": True,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 1000),
    },
    "Xception": {
        "module": "kmodels.models.xception",
        "model_cls": "Xception",
        "model_type": "backbone",
        "init_kwargs": {
            "weights": None,
            "input_shape": (71, 71, 3),
            "include_top": True,
        },
        "input_shape": (2, 71, 71, 3),
        "expected_output_shape": (2, 1000),
    },
    "DETRResNet50": {
        "module": "kmodels.models.detr",
        "model_cls": "DETRResNet50",
        "model_type": "object_detection",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "num_classes": 92,
            "num_queries": 10,
            "include_normalization": False,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": {
            "logits": (2, 10, 92),
            "pred_boxes": (2, 10, 4),
        },
    },
    "RFDETRNano": {
        "module": "kmodels.models.rf_detr",
        "model_cls": "RFDETRNano",
        "model_type": "object_detection",
        "init_kwargs": {
            "weights": None,
            "num_queries": 10,
            "num_classes": 91,
            "input_shape": (256, 256, 3),
        },
        "input_shape": (2, 256, 256, 3),
        "expected_output_shape": {
            "pred_logits": (2, 10, 91),
            "pred_boxes": (2, 10, 4),
        },
    },
    "DeepLabV3ResNet50": {
        "module": "kmodels.models.deeplabv3",
        "model_cls": "DeepLabV3ResNet50",
        "model_type": "segmentation",
        "init_kwargs": {
            "weights": None,
            "input_shape": (64, 64, 3),
            "num_classes": 21,
        },
        "input_shape": (2, 64, 64, 3),
        "expected_output_shape": (2, 64, 64, 21),
    },
    "EoMT_Small": {
        "module": "kmodels.models.eomt",
        "model_cls": "EoMT_Small",
        "model_type": "segmentation",
        "init_kwargs": {
            "weights": None,
            "input_shape": (64, 64, 3),
            "num_queries": 100,
            "num_labels": 133,
        },
        "input_shape": (2, 64, 64, 3),
        "expected_output_shape": {
            "class_logits": (2, 100, 134),
            "mask_logits": (2, 100, 16, 16),
        },
    },
    "SegFormerB0": {
        "module": "kmodels.models.segformer",
        "model_cls": "SegFormerB0",
        "model_type": "segmentation",
        "init_kwargs": {
            "weights": None,
            "input_shape": (32, 32, 3),
            "num_classes": 150,
        },
        "input_shape": (2, 32, 32, 3),
        "expected_output_shape": (2, 32, 32, 150),
    },
    "SAM_ViT_Base": {
        "module": "kmodels.models.sam",
        "model_cls": "SAM_ViT_Base",
        "model_type": "promptable_segmentation",
        "init_kwargs": {
            "weights": None,
            "input_shape": (64, 64, 3),
        },
        "input_factory": "sam_input",
        "input_factory_kwargs": {"image_size": 64},
        "expected_output_shape": {
            "pred_masks": None,  # dynamic shape based on prompts
            "iou_scores": None,
        },
    },
    "Sam2Tiny": {
        "module": "kmodels.models.sam2",
        "model_cls": "Sam2Tiny",
        "model_type": "promptable_segmentation",
        "init_kwargs": {
            "weights": None,
            "input_shape": (128, 128, 3),
        },
        "input_factory": "sam_input",
        "input_factory_kwargs": {"image_size": 128},
        "expected_output_shape": {
            "pred_masks": None,
            "iou_scores": None,
            "object_score_logits": None,
        },
    },
    "ClipVitBase32": {
        "module": "kmodels.models.clip",
        "model_cls": "ClipVitBase32",
        "model_type": "vlm",
        "init_kwargs": {
            "weights": None,
            "input_shape": (64, 64, 3),
        },
        "input_factory": "clip_input",
        "input_factory_kwargs": {"image_size": 64},
        "expected_output_shape": {
            "image_logits": (2, 2),
            "text_logits": (2, 2),
        },
    },
    "SigLIPBaseP16": {
        "module": "kmodels.models.siglip",
        "model_cls": "SigLIPBaseP16",
        "model_type": "vlm",
        "init_kwargs": {
            "weights": None,
            "input_shape": (64, 64, 3),
        },
        "input_factory": "siglip_input",
        "input_factory_kwargs": {"image_size": 64},
        "expected_output_shape": {
            "image_logits": (2, 2),
            "text_logits": (2, 2),
        },
    },
    "SigLIP2BaseP16": {
        "module": "kmodels.models.siglip2",
        "model_cls": "SigLIP2BaseP16",
        "model_type": "vlm",
        "init_kwargs": {
            "weights": None,
            "input_shape": (64, 64, 3),
        },
        "input_factory": "siglip_input",
        "input_factory_kwargs": {"image_size": 64},
        "expected_output_shape": {
            "image_logits": (2, 2),
            "text_logits": (2, 2),
        },
    },
}


def get_all_model_ids():
    return list(MODEL_TEST_CONFIGS.keys())


def get_models_by_type(model_type):
    return {
        k: v for k, v in MODEL_TEST_CONFIGS.items() if v["model_type"] == model_type
    }


def import_model_class(config):
    import importlib

    module = importlib.import_module(config["module"])
    return getattr(module, config["model_cls"])


def create_test_input(config, batch_size=2):
    from tests.fixtures import dummy_inputs

    if "input_factory" in config:
        factory_fn = getattr(dummy_inputs, config["input_factory"])
        factory_kwargs = config.get("input_factory_kwargs", {})
        return factory_fn(batch_size=batch_size, **factory_kwargs)

    from keras import ops

    shape = config["input_shape"]
    return ops.ones(shape)
