import keras
from keras import layers, ops

from kvmm.models.yolo.blocks import c3_block, conv_block, sppf_block
from kvmm.models.yolo.head import detect_head
from kvmm.models.yolo.utils import scale_channels, scale_depth
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import YOLOV5_MODEL_CONFIG, YOLOV5_WEIGHTS_CONFIG
from kvmm.model_registry import register_model

def build_backbone_and_neck(images_input, width_multiple, depth_multiple, data_format):
    """
    Build the backbone and neck architecture of YOLOv5.
    
    This function constructs the feature extraction backbone of YOLOv5, which consists of
    a series of convolutional blocks and C3 blocks that progressively downsample the input
    while increasing the number of channels. The backbone extracts multi-scale features
    that are essential for object detection at different scales.
    
    The architecture follows the YOLOv5 design:
    - Initial conv blocks for feature extraction and downsampling
    - C3 blocks for efficient feature learning with residual connections
    - SPPF block for spatial pyramid pooling to capture multi-scale context
    - Three feature maps (P3, P4, P5) are extracted at different scales for the neck
    
    Args:
        images_input (keras.KerasTensor): Input tensor containing batch of images with shape
            [batch_size, height, width, channels] for 'channels_last' format or
            [batch_size, channels, height, width] for 'channels_first' format.
        width_multiple (float): Scaling factor for channel width. Controls the number
            of channels in each layer (e.g., 0.5 for YOLOv5s, 1.0 for YOLOv5m).
        depth_multiple (float): Scaling factor for layer depth. Controls the number
            of repeated blocks in C3 layers (e.g., 0.33 for YOLOv5s, 1.0 for YOLOv5m).
        data_format (str): Data format specification, either 'channels_last' (NHWC)
            or 'channels_first' (NCHW). Determines the arrangement of tensor dimensions.
    
    Returns:
        tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]: A tuple containing 
        three feature tensors:
            - p3_features: Feature map from P3 level (1/8 scale) with shape corresponding
              to 256 * width_multiple channels
            - p4_features: Feature map from P4 level (1/16 scale) with shape corresponding
              to 512 * width_multiple channels  
            - p5_features: Feature map from P5 level (1/32 scale) with shape corresponding
              to 1024 * width_multiple channels
            
            These feature maps are used by the neck network for feature fusion and
            final detection head processing.
    """
    inputs = conv_block(
        images_input,
        scale_channels(64, width_multiple),
        kernel_size=6,
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

    inputs = c3_block(
        inputs,
        scale_channels(128, width_multiple),
        n=scale_depth(3, depth_multiple),
        shortcut=True,
        data_format=data_format,
        name_prefix="c3_block_2",
    )

    inputs = conv_block(
        inputs,
        scale_channels(256, width_multiple),
        kernel_size=3,
        strides=2,
        data_format=data_format,
        name_prefix="conv_block_3",
    )

    inputs = c3_block(
        inputs,
        scale_channels(256, width_multiple),
        n=scale_depth(6, depth_multiple),
        shortcut=True,
        data_format=data_format,
        name_prefix="c3_block_4",
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

    inputs = c3_block(
        inputs,
        scale_channels(512, width_multiple),
        n=scale_depth(9, depth_multiple),
        shortcut=True,
        data_format=data_format,
        name_prefix="c3_block_6",
    )
    p4_features = inputs

    inputs = conv_block(
        inputs,
        scale_channels(1024, width_multiple),
        kernel_size=3,
        strides=2,
        data_format=data_format,
        name_prefix="conv_block_7",
    )

    inputs = c3_block(
        inputs,
        scale_channels(1024, width_multiple),
        n=scale_depth(3, depth_multiple),
        shortcut=True,
        data_format=data_format,
        name_prefix="c3_block_8",
    )

    inputs = sppf_block(
        inputs,
        scale_channels(1024, width_multiple),
        kernel_size=5,
        data_format=data_format,
        name_prefix="sppf_block_9",
    )
    p5_features = inputs

    return p3_features, p4_features, p5_features


def build_fpn(
    p3_features, p4_features, p5_features, width_multiple, depth_multiple, data_format
):
    """
    Build Feature Pyramid Network (FPN) - Top-down pathway for YOLOv5.
    
    This function implements the top-down pathway of the Feature Pyramid Network,
    which is a crucial component of the YOLOv5 neck architecture. The FPN enables
    the model to detect objects at multiple scales by combining high-resolution,
    low-level features with low-resolution, high-level semantic features.
    
    The FPN process:
    1. Reduces P5 features to 512 channels and upsamples by 2x
    2. Concatenates upsampled P5 with P4 features along channel dimension
    3. Processes concatenated features through C3 block for feature refinement
    4. Reduces processed P4 features to 256 channels and upsamples by 2x
    5. Concatenates upsampled P4 with P3 features along channel dimension
    6. Processes final concatenated features through C3 block
    
    This top-down information flow allows higher-level semantic information from
    deeper layers to enhance the feature representation at shallower layers,
    improving detection accuracy for objects of various sizes.
    
    Args:
        p3_features (keras.KerasTensor): Feature map from P3 level (1/8 scale) of the
            backbone network with shape corresponding to 256 * width_multiple channels.
        p4_features (keras.KerasTensor): Feature map from P4 level (1/16 scale) of the
            backbone network with shape corresponding to 512 * width_multiple channels.
        p5_features (keras.KerasTensor): Feature map from P5 level (1/32 scale) of the
            backbone network with shape corresponding to 1024 * width_multiple channels.
        width_multiple (float): Scaling factor for channel width. Controls the number
            of channels in each layer (e.g., 0.5 for YOLOv5s, 1.0 for YOLOv5m).
        depth_multiple (float): Scaling factor for layer depth. Controls the number
            of repeated blocks in C3 layers (e.g., 0.33 for YOLOv5s, 1.0 for YOLOv5m).
        data_format (str): Data format specification, either 'channels_last' (NHWC)
            or 'channels_first' (NCHW). Determines the arrangement of tensor dimensions
            and concatenation axis.
    
    Returns:
        tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]: A tuple containing
        three processed feature tensors:
            - p3_out: Enhanced P3 features after FPN processing with shape corresponding
              to 256 * width_multiple channels. Ready for detection head at 1/8 scale.
            - p4_reduced: Intermediate P4 features reduced to 256 * width_multiple channels.
              Used as input for the bottom-up pathway (PAN).
            - p5_reduced: Intermediate P5 features reduced to 512 * width_multiple channels.
              Used as input for the bottom-up pathway (PAN).
    """
    if data_format == "channels_last":
        concat_axis = -1
    else:
        concat_axis = 1

    # Top-down pathway
    p5_reduced = conv_block(
        p5_features,
        scale_channels(512, width_multiple),
        kernel_size=1,
        strides=1,
        data_format=data_format,
        name_prefix="conv_block_10",
    )

    p5_upsampled = layers.UpSampling2D(
        size=2, data_format=data_format, interpolation="nearest", name="upsample_1"
    )(p5_reduced)

    p4_concat = layers.Concatenate(axis=concat_axis, name="concat_1")(
        [p5_upsampled, p4_features]
    )

    p4_processed = c3_block(
        p4_concat,
        scale_channels(512, width_multiple),
        n=scale_depth(3, depth_multiple),
        shortcut=False,
        data_format=data_format,
        name_prefix="c3_block_13",
    )

    p4_reduced = conv_block(
        p4_processed,
        scale_channels(256, width_multiple),
        kernel_size=1,
        strides=1,
        data_format=data_format,
        name_prefix="conv_block_14",
    )

    p4_upsampled = layers.UpSampling2D(
        size=2, data_format=data_format, interpolation="nearest", name="upsample_2"
    )(p4_reduced)

    p3_concat = layers.Concatenate(axis=concat_axis, name="concat_2")(
        [p4_upsampled, p3_features]
    )

    p3_out = c3_block(
        p3_concat,
        scale_channels(256, width_multiple),
        n=scale_depth(3, depth_multiple),
        shortcut=False,
        data_format=data_format,
        name_prefix="c3_block_17",
    )

    return p3_out, p4_reduced, p5_reduced


def build_pan(
    p3_out, p4_reduced, p5_reduced, width_multiple, depth_multiple, data_format
):
    """
    Build Path Aggregation Network (PAN) - Bottom-up pathway for YOLOv5.
    
    This function implements the bottom-up pathway of the Path Aggregation Network,
    which is the second part of the YOLOv5 neck architecture after the FPN. The PAN
    enhances feature propagation by adding a bottom-up path that allows low-level
    features to be directly propagated to higher levels, improving localization
    accuracy for small objects.
    
    The PAN process:
    1. Downsamples P3 features by 2x using stride-2 convolution
    2. Concatenates downsampled P3 with reduced P4 features from FPN
    3. Processes concatenated features through C3 block for feature fusion
    4. Downsamples processed P4 features by 2x using stride-2 convolution  
    5. Concatenates downsampled P4 with reduced P5 features from FPN
    6. Processes final concatenated features through C3 block
    
    This bottom-up information flow creates stronger feature pyramids by shortening
    the information path between lower and higher pyramid levels, which is especially
    beneficial for detecting small objects that require fine-grained localization.
    
    Args:
        p3_out (keras.KerasTensor): Enhanced P3 features from FPN top-down pathway
            with shape corresponding to 256 * width_multiple channels at 1/8 scale.
        p4_reduced (keras.KerasTensor): Intermediate P4 features from FPN, reduced to
            256 * width_multiple channels at 1/16 scale.
        p5_reduced (keras.KerasTensor): Intermediate P5 features from FPN, reduced to
            512 * width_multiple channels at 1/32 scale.
        width_multiple (float): Scaling factor for channel width. Controls the number
            of channels in each layer (e.g., 0.5 for YOLOv5s, 1.0 for YOLOv5m).
        depth_multiple (float): Scaling factor for layer depth. Controls the number
            of repeated blocks in C3 layers (e.g., 0.33 for YOLOv5s, 1.0 for YOLOv5m).
        data_format (str): Data format specification, either 'channels_last' (NHWC)
            or 'channels_first' (NCHW). Determines the arrangement of tensor dimensions
            and concatenation axis.
    
    Returns:
        list[keras.KerasTensor]: A list containing three final feature tensors ready
        for detection heads:
            - p3_out: Final P3 features (unchanged from input) with 256 * width_multiple
              channels at 1/8 scale. Used for detecting small objects.
            - p4_out: Enhanced P4 features after PAN processing with 512 * width_multiple
              channels at 1/16 scale. Used for detecting medium objects.
            - p5_out: Enhanced P5 features after PAN processing with 1024 * width_multiple
              channels at 1/32 scale. Used for detecting large objects.
    """
    if data_format == "channels_last":
        concat_axis = -1
    else:
        concat_axis = 1

    # Bottom-up pathway
    p3_downsampled = conv_block(
        p3_out,
        scale_channels(256, width_multiple),
        kernel_size=3,
        strides=2,
        data_format=data_format,
        name_prefix="conv_block_18",
    )

    p4_final_concat = layers.Concatenate(axis=concat_axis, name="concat_3")(
        [p3_downsampled, p4_reduced]
    )

    p4_out = c3_block(
        p4_final_concat,
        scale_channels(512, width_multiple),
        n=scale_depth(3, depth_multiple),
        shortcut=False,
        data_format=data_format,
        name_prefix="c3_block_20",
    )

    p4_downsampled = conv_block(
        p4_out,
        scale_channels(512, width_multiple),
        kernel_size=3,
        strides=2,
        data_format=data_format,
        name_prefix="conv_block_21",
    )

    p5_final_concat = layers.Concatenate(axis=concat_axis, name="concat_4")(
        [p4_downsampled, p5_reduced]
    )

    p5_out = c3_block(
        p5_final_concat,
        scale_channels(1024, width_multiple),
        n=scale_depth(3, depth_multiple),
        shortcut=False,
        data_format=data_format,
        name_prefix="c3_block_23",
    )

    return [p3_out, p4_out, p5_out]


class YOLOv5(keras.Model):
    def __init__(
        self,
        input_shape=(None, None, 3),
        nc=80,
        data_format="channels_last",
        max_boxes=300,
        depth_multiple=0.33,
        width_multiple=0.50,
        input_tensor=None,
        training=True,  # Add training flag
        name="YOLOv5",
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

        # Define 3 separate inputs
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

        # Set up inputs
        if training:
            inputs = {
                "images": images_input,
                "bbox": bbox_input,
                "labels": labels_input,
            }
        else:
            inputs = images_input

        # Build architecture using modular functions
        p3_features, p4_features, p5_features = build_backbone_and_neck(
            images_input, width_multiple, depth_multiple, data_format
        )

        p3_out, p4_reduced, p5_reduced = build_fpn(
            p3_features,
            p4_features,
            p5_features,
            width_multiple,
            depth_multiple,
            data_format,
        )

        feature_maps = build_pan(
            p3_out, p4_reduced, p5_reduced, width_multiple, depth_multiple, data_format
        )

        # Detection head
        detection_outputs = detect_head(
            feature_maps,
            nc=nc,
            reg_max=16,
            data_format=data_format,
            name_prefix="detect_head_24",
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

        # Store configuration attributes
        self.nc = nc
        self.data_format = data_format
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
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
                "training": getattr(self, "training_mode", True),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_model
def YoloV5n(
    weights="coco",
    input_tensor=None,
    nc=80,
    input_shape=(None, None, 3),
    training=False,
    name="YoloV5n",
    **kwargs,
):
    model = YOLOv5(
        **YOLOV5_MODEL_CONFIG["YoloV5n"],
        input_shape=input_shape,
        nc=nc,
        input_tensor=input_tensor,
        training=training,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(YOLOV5_WEIGHTS_CONFIG):
        load_weights_from_config("YoloV5n", weights, model, YOLOV5_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model

@register_model
def YoloV5s(
    weights="coco",
    input_tensor=None,
    nc=80,
    input_shape=(None, None, 3),
    training=False,
    name="YoloV5s",
    **kwargs,
):
    model = YOLOv5(
        **YOLOV5_MODEL_CONFIG["YoloV5s"],
        input_shape=input_shape,
        nc=nc,
        input_tensor=input_tensor,
        training=training,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(YOLOV5_WEIGHTS_CONFIG):
        load_weights_from_config("YoloV5s", weights, model, YOLOV5_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model

@register_model
def YoloV5m(
    weights="coco",
    input_tensor=None,
    nc=80,
    input_shape=(None, None, 3),
    training=False,
    name="YoloV5m",
    **kwargs,
):
    model = YOLOv5(
        **YOLOV5_MODEL_CONFIG["YoloV5m"],
        input_shape=input_shape,
        nc=nc,
        input_tensor=input_tensor,
        training=training,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(YOLOV5_WEIGHTS_CONFIG):
        load_weights_from_config("YoloV5m", weights, model, YOLOV5_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model

@register_model
def YoloV5l(
    weights="coco",
    input_tensor=None,
    nc=80,
    input_shape=(None, None, 3),
    training=False,
    name="YoloV5l",
    **kwargs,
):
    model = YOLOv5(
        **YOLOV5_MODEL_CONFIG["YoloV5l"],
        input_shape=input_shape,
        nc=nc,
        input_tensor=input_tensor,
        training=training,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(YOLOV5_WEIGHTS_CONFIG):
        load_weights_from_config("YoloV5l", weights, model, YOLOV5_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model

@register_model
def YoloV5x(
    weights="coco",
    input_tensor=None,
    nc=80,
    input_shape=(None, None, 3),
    training=False,
    name="YoloV5x",
    **kwargs,
):
    model = YOLOv5(
        **YOLOV5_MODEL_CONFIG["YoloV5x"],
        input_shape=input_shape,
        nc=nc,
        input_tensor=input_tensor,
        training=training,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(YOLOV5_WEIGHTS_CONFIG):
        load_weights_from_config("YoloV5x", weights, model, YOLOV5_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
