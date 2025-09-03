import keras
from keras import layers, ops

from kvmm.models.yolo.blocks import conv_block, sppf_block, c2f_block
from kvmm.models.yolo.head import detect_head
from kvmm.models.yolo.utils import scale_channels, scale_depth
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import YOLOV8_MODEL_CONFIG, YOLOV8_WEIGHTS_CONFIG


def build_backbone_and_neck(images_input, width_multiple, depth_multiple, data_format, max_channels=1024):
    """
    Build the backbone and neck architecture of YOLOv8.

    Args:
        images_input: Input tensor for images
        width_multiple: Width scaling factor
        depth_multiple: Depth scaling factor
        data_format: Data format ('channels_last' or 'channels_first')
        max_channels: Maximum number of channels allowed (used for memory optimization)

    Returns:
        tuple: (p3_features, p4_features, p5_features) - Feature maps at different scales
    """
    inputs = conv_block(
        images_input,
        scale_channels(64, width_multiple),
        kernel_size=3,
        strides=2,
        data_format=data_format,
        name_prefix="conv_block_0",
    )

    inputs = conv_block(
        inputs,
        scale_channels(128, width_multiple),
        kernel_size=3,
        strides=2,
        data_format=data_format,
        name_prefix="conv_block_1",
    )

    inputs = c2f_block(
        inputs,
        scale_channels(128, width_multiple),
        n=scale_depth(3, depth_multiple),
        shortcut=True,
        data_format=data_format,
        name_prefix="c2f_block_2",
    )

    inputs = conv_block(
        inputs,
        scale_channels(256, width_multiple),
        kernel_size=3,
        strides=2,
        data_format=data_format,
        name_prefix="conv_block_3",
    )

    inputs = c2f_block(
        inputs,
        scale_channels(256, width_multiple),
        n=scale_depth(6, depth_multiple),
        shortcut=True,
        data_format=data_format,
        name_prefix="c2f_block_4",
    )
    p3_features = inputs

    inputs = conv_block(
        inputs,
        scale_channels(512, width_multiple),
        kernel_size=3,
        strides=2,
        data_format=data_format,
        name_prefix="conv_block_5",
    )

    inputs = c2f_block(
        inputs,
        scale_channels(512, width_multiple),
        n=scale_depth(6, depth_multiple),
        shortcut=True,
        data_format=data_format,
        name_prefix="c2f_block_6",
    )
    p4_features = inputs
    inputs = conv_block(
        inputs,
        scale_channels(512, width_multiple),
        kernel_size=3,
        strides=2,
        data_format=data_format,
        name_prefix="conv_block_7",
    )

    inputs = c2f_block(
        inputs,
        scale_channels(512, width_multiple),
        n=scale_depth(3, depth_multiple),
        shortcut=True,
        data_format=data_format,
        name_prefix="c2f_block_8",
    )

    inputs = sppf_block(
        inputs,
        scale_channels(512, width_multiple),
        kernel_size=5,
        data_format=data_format,
        name_prefix="sppf_block_9",
    )
    p5_features = inputs

    return p3_features, p4_features, p5_features


def build_fpn(
    p3_features, p4_features, p5_features, width_multiple, depth_multiple, data_format, max_channels=1024
):
    """
    Build Feature Pyramid Network (FPN) - Top-down pathway for YOLOv8.

    Args:
        p3_features: P3 feature map from backbone
        p4_features: P4 feature map from backbone
        p5_features: P5 feature map from backbone
        width_multiple: Width scaling factor
        depth_multiple: Depth scaling factor
        data_format: Data format ('channels_last' or 'channels_first')
        max_channels: Maximum number of channels allowed

    Returns:
        tuple: (p3_out, p4_out) - Processed feature maps
    """
    if data_format == "channels_last":
        concat_axis = -1
    else:
        concat_axis = 1

    p5_upsampled = layers.UpSampling2D(
        size=2, data_format=data_format, interpolation="nearest", name="upsample_10"
    )(p5_features)

    p4_concat = layers.Concatenate(axis=concat_axis, name="concat_11")(
        [p5_upsampled, p4_features]
    )

    p4_processed = c2f_block(
        p4_concat,
        scale_channels(512, width_multiple),
        n=scale_depth(3, depth_multiple),
        shortcut=False,
        data_format=data_format,
        name_prefix="c2f_block_12",
    )

    p4_upsampled = layers.UpSampling2D(
        size=2, data_format=data_format, interpolation="nearest", name="upsample_13"
    )(p4_processed)

    p3_concat = layers.Concatenate(axis=concat_axis, name="concat_14")(
        [p4_upsampled, p3_features]
    )

    p3_out = c2f_block(
        p3_concat,
        scale_channels(256, width_multiple),
        n=scale_depth(3, depth_multiple),
        shortcut=False,
        data_format=data_format,
        name_prefix="c2f_block_15",
    )

    return p3_out, p4_processed, p5_features


def build_pan(
    p3_out, p4_processed, p5_features, width_multiple, depth_multiple, data_format, max_channels=1024
):
    """
    Build Path Aggregation Network (PAN) - Bottom-up pathway for YOLOv8.

    Args:
        p3_out: P3 output from FPN
        p4_processed: P4 processed features from FPN
        p5_features: P5 features from backbone
        width_multiple: Width scaling factor
        depth_multiple: Depth scaling factor
        data_format: Data format ('channels_last' or 'channels_first')
        max_channels: Maximum number of channels allowed

    Returns:
        list: [p3_out, p4_out, p5_out] - Final feature maps for detection heads
    """
    if data_format == "channels_last":
        concat_axis = -1
    else:
        concat_axis = 1

    p3_downsampled = conv_block(
        p3_out,
        scale_channels(256, width_multiple),
        kernel_size=3,
        strides=2,
        data_format=data_format,
        name_prefix="conv_block_16",
    )

    p4_final_concat = layers.Concatenate(axis=concat_axis, name="concat_17")(
        [p3_downsampled, p4_processed]
    )

    p4_out = c2f_block(
        p4_final_concat,
        scale_channels(512, width_multiple),
        n=scale_depth(3, depth_multiple),
        shortcut=False,
        data_format=data_format,
        name_prefix="c2f_block_18",
    )

    p4_downsampled = conv_block(
        p4_out,
        scale_channels(512, width_multiple),
        kernel_size=3,
        strides=2,
        data_format=data_format,
        name_prefix="conv_block_19",
    )

    p5_final_concat = layers.Concatenate(axis=concat_axis, name="concat_20")(
        [p4_downsampled, p5_features]
    )

    p5_out = c2f_block(
        p5_final_concat,
        scale_channels(512, width_multiple),
        n=scale_depth(3, depth_multiple),
        shortcut=False,
        data_format=data_format,
        name_prefix="c2f_block_21",
    )

    return [p3_out, p4_out, p5_out]


class YOLOv8(keras.Model):
    def __init__(
        self,
        input_shape=(None, None, 3),
        nc=80,
        data_format="channels_last",
        max_boxes=300,
        depth_multiple=0.33,
        width_multiple=0.50,
        max_channels=1024,
        input_tensor=None,
        training=True,
        name="YOLOv8",
        **kwargs,
    ):
        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                "The `data_format` argument should be one of 'channels_last' or 'channels_first'. "
                f"Received: data_format={data_format}"
            )

        if input_shape is None:
            image_size = 640
            channels = 3
        else:
            if len(input_shape) == 3:
                if data_format == "channels_first":
                    channels, image_size, _ = input_shape
                else:
                    image_size, _, channels = input_shape
            else:
                image_size = 640
                channels = 3

        if data_format == "channels_first":
            image_input_shape = [channels, image_size, image_size]
        else:
            image_input_shape = [image_size, image_size, channels]

        # Define inputs
        if isinstance(input_tensor, dict):
            images_input = input_tensor.get("images") or layers.Input(
                shape=image_input_shape, name="images"
            )
            if training:
                bbox_input = input_tensor.get("bbox") or layers.Input(
                    shape=[max_boxes, 4],
                    name="bbox",  # [x1, y1, x2, y2]
                )
                labels_input = input_tensor.get("labels") or layers.Input(
                    shape=[max_boxes],
                    name="labels",  # class labels
                )
        else:
            images_input = layers.Input(shape=image_input_shape, name="images")
            if training:
                bbox_input = layers.Input(shape=[max_boxes, 4], name="bbox")
                labels_input = layers.Input(shape=[max_boxes], name="labels")

        if training:
            inputs = {
                "images": images_input,
                "bbox": bbox_input,
                "labels": labels_input,
            }
        else:
            inputs = images_input

        p3_features, p4_features, p5_features = build_backbone_and_neck(
            images_input, width_multiple, depth_multiple, data_format, max_channels=max_channels
        )

        p3_out, p4_processed, p5_features = build_fpn(
            p3_features,
            p4_features,
            p5_features,
            width_multiple,
            depth_multiple,
            data_format,
            max_channels=max_channels,
        )

        feature_maps = build_pan(
            p3_out, p4_processed, p5_features, width_multiple, depth_multiple, data_format, max_channels=max_channels
        )

        detection_outputs = detect_head(
            feature_maps,
            nc=nc,
            reg_max=16,
            data_format=data_format,
            name_prefix="detect_head_22",
        )

        # Handle outputs based on training mode
        if training:
            # Create valid mask: valid if label >= 0 (assuming -1 for padding)
            valid_mask = ops.cast(labels_input >= 0, "float32")

            outputs = {
                "predictions": detection_outputs,
                "targets": {
                    "boxes": bbox_input,
                    "labels": labels_input,
                    "valid_mask": valid_mask,
                },
            }
        else:
            outputs = detection_outputs

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.nc = nc
        self.data_format = data_format
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.max_channels = max_channels
        self.training_mode = training

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": getattr(self, "input_shape", None),
                "nc": getattr(self, "nc", 80),
                "data_format": getattr(self, "data_format", "channels_last"),
                "depth_multiple": getattr(self, "depth_multiple", 0.33),
                "width_multiple": getattr(self, "width_multiple", 0.50),
                "max_channels": getattr(self, "max_channels", 1024),
                "training": getattr(self, "training_mode", True),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def YoloV8n(
    weights="coco",
    input_tensor=None,
    nc=80,
    input_shape=(None, None, 3),
    training=False,
    name="YoloV8n",
    **kwargs,
):
    model = YOLOv8(
        **YOLOV8_MODEL_CONFIG["YoloV8n"],
        input_shape=input_shape,
        nc=nc,
        input_tensor=input_tensor,
        training=training,
        name=name,
        **kwargs,
    )
    if weights in get_all_weight_names(YOLOV8_WEIGHTS_CONFIG):
        load_weights_from_config("YoloV5n", weights, model, YOLOV8_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")
    return model


def YoloV8s(
    weights="coco",
    input_tensor=None,
    nc=80,
    input_shape=(None, None, 3),
    training=False,
    name="YoloV8s",
    **kwargs,
):
    model = YOLOv8(
        **YOLOV8_MODEL_CONFIG["YoloV8s"],
        input_shape=input_shape,
        nc=nc,
        input_tensor=input_tensor,
        training=training,
        name=name,
        **kwargs,
    )
    if weights in get_all_weight_names(YOLOV8_WEIGHTS_CONFIG):
        load_weights_from_config("YoloV5s", weights, model, YOLOV8_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")
    return model


def YoloV8m(
    weights="coco",
    input_tensor=None,
    nc=80,
    input_shape=(None, None, 3),
    training=False,
    name="YoloV8m",
    **kwargs,
):
    model = YOLOv8(
        **YOLOV8_MODEL_CONFIG["YoloV8m"],
        input_shape=input_shape,
        nc=nc,
        input_tensor=input_tensor,
        training=training,
        name=name,
        **kwargs,
    )
    if weights in get_all_weight_names(YOLOV8_WEIGHTS_CONFIG):
        load_weights_from_config("YoloV5m", weights, model, YOLOV8_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")
    return model


def YoloV8l(
    weights="coco",
    input_tensor=None,
    nc=80,
    input_shape=(None, None, 3),
    training=False,
    name="YoloV8l",
    **kwargs,
):
    model = YOLOv8(
        **YOLOV8_MODEL_CONFIG["YoloV8l"],
        input_shape=input_shape,
        nc=nc,
        input_tensor=input_tensor,
        training=training,
        name=name,
        **kwargs,
    )
    if weights in get_all_weight_names(YOLOV8_WEIGHTS_CONFIG):
        load_weights_from_config("YoloV5l", weights, model, YOLOV8_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")
    return model


def YoloV8x(
    weights="coco",
    input_tensor=None,
    nc=80,
    input_shape=(None, None, 3),
    training=False,
    name="YoloV8x",
    **kwargs,
):
    model = YOLOv8(
        **YOLOV8_MODEL_CONFIG["YoloV8x"],
        input_shape=input_shape,
        nc=nc,
        input_tensor=input_tensor,
        training=training,
        name=name,
        **kwargs,
    )
    if weights in get_all_weight_names(YOLOV8_WEIGHTS_CONFIG):
        load_weights_from_config("YoloV5x", weights, model, YOLOV8_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")
    return model