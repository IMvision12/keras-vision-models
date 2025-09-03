import keras
from keras import layers, ops

from kvmm.models.yolo.blocks import conv_block, sppf_block, c2f_block
from kvmm.models.yolo.head import detect_head
from kvmm.models.yolo.utils import scale_channels, scale_depth
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import YOLOV8_MODEL_CONFIG, YOLOV8_WEIGHTS_CONFIG
from kvmm.model_registry import register_model

def build_backbone_and_neck(images_input, width_multiple, depth_multiple, data_format):
    """
    Build the backbone and neck architecture of YOLOv8.
    
    This function constructs the feature extraction backbone of YOLOv8, which represents
    an evolution from YOLOv5 with improved architectural components. The backbone consists
    of convolutional blocks and C2f blocks (Cross Stage Partial with 2 convolutions + more
    shortcut connections) that progressively downsample the input while increasing channel
    depth for hierarchical feature learning.
    
    Key improvements in YOLOv8 architecture:
    - C2f blocks replace C3 blocks for better gradient flow and feature reuse
    - More balanced channel progression (512->512 instead of 512->1024 at the end)
    - Enhanced spatial pyramid pooling with SPPF for multi-scale context
    - Three feature maps (P3, P4, P5) extracted at different scales for the neck
    
    The architecture follows this pattern:
    - Initial conv blocks for feature extraction and progressive downsampling
    - C2f blocks for efficient feature learning with enhanced skip connections
    - SPPF block for spatial pyramid pooling to capture multi-scale context
    - Feature extraction at 1/8, 1/16, and 1/32 scales for multi-scale detection
    
    Args:
        images_input (keras.KerasTensor): Input tensor containing batch of images with shape
            [batch_size, height, width, channels] for 'channels_last' format or
            [batch_size, channels, height, width] for 'channels_first' format.
        width_multiple (float): Scaling factor for channel width. Controls the number
            of channels in each layer (e.g., 0.25 for YOLOv8n, 0.5 for YOLOv8s, 1.0 for YOLOv8m).
        depth_multiple (float): Scaling factor for layer depth. Controls the number
            of repeated blocks in C2f layers (e.g., 0.33 for YOLOv8n, 0.5 for YOLOv8s, 1.0 for YOLOv8m).
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
              to 512 * width_multiple channels
              
            These feature maps are used by the neck network (FPN + PAN) for feature
            fusion and final detection head processing.
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
    p3_features, p4_features, p5_features, width_multiple, depth_multiple, data_format
):
    """
    Build Feature Pyramid Network (FPN) - Top-down pathway for YOLOv8.
    
    This function implements the top-down pathway of the Feature Pyramid Network,
    which is a crucial component of the YOLOv8 neck architecture. The FPN enables
    the model to detect objects at multiple scales by combining high-resolution,
    low-level features with low-resolution, high-level semantic features.
    
    Key differences from YOLOv5 FPN:
    - Uses C2f blocks instead of C3 blocks for better feature fusion
    - Simplified architecture with direct upsampling and concatenation
    - More efficient feature flow without intermediate channel reduction
    - Enhanced gradient propagation through C2f block design
    
    The FPN process:
    1. Upsamples P5 features by 2x using nearest neighbor interpolation
    2. Concatenates upsampled P5 with P4 features along channel dimension
    3. Processes concatenated features through C2f block for feature refinement
    4. Upsamples processed P4 features by 2x using nearest neighbor interpolation
    5. Concatenates upsampled P4 with P3 features along channel dimension
    6. Processes final concatenated features through C2f block
    
    This top-down information flow allows higher-level semantic information from
    deeper layers to enhance the feature representation at shallower layers,
    improving detection accuracy for objects of various sizes.
    
    Args:
        p3_features (keras.KerasTensor): Feature map from P3 level (1/8 scale) of the
            backbone network with shape corresponding to 256 * width_multiple channels.
        p4_features (keras.KerasTensor): Feature map from P4 level (1/16 scale) of the
            backbone network with shape corresponding to 512 * width_multiple channels.
        p5_features (keras.KerasTensor): Feature map from P5 level (1/32 scale) of the
            backbone network with shape corresponding to 512 * width_multiple channels.
        width_multiple (float): Scaling factor for channel width. Controls the number
            of channels in each layer (e.g., 0.25 for YOLOv8n, 0.5 for YOLOv8s, 1.0 for YOLOv8m).
        depth_multiple (float): Scaling factor for layer depth. Controls the number
            of repeated blocks in C2f layers (e.g., 0.33 for YOLOv8n, 0.5 for YOLOv8s, 1.0 for YOLOv8m).
        data_format (str): Data format specification, either 'channels_last' (NHWC)
            or 'channels_first' (NCHW). Determines the arrangement of tensor dimensions
            and concatenation axis.
    
    Returns:
        tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]: A tuple containing
        three processed feature tensors:
            - p3_out: Enhanced P3 features after FPN processing with shape corresponding
              to 256 * width_multiple channels. Ready for detection head at 1/8 scale.
            - p4_processed: Enhanced P4 features after FPN processing with shape corresponding
              to 512 * width_multiple channels. Used as input for the bottom-up pathway (PAN).
            - p5_features: Original P5 features (unchanged) with shape corresponding to

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
    p3_out, p4_processed, p5_features, width_multiple, depth_multiple, data_format
):
    """
    Build Path Aggregation Network (PAN) - Bottom-up pathway for YOLOv8.
    
    This function implements the bottom-up pathway of the Path Aggregation Network,
    which is the second part of the YOLOv8 neck architecture after the FPN. The PAN
    enhances feature propagation by adding a bottom-up path that allows low-level
    features to be directly propagated to higher levels, improving localization
    accuracy for small objects.
    
    Key differences from YOLOv5 PAN:
    - Uses C2f blocks instead of C3 blocks for better gradient flow and feature reuse
    - More balanced channel progression (512 channels for P5 instead of 1024)
    - Enhanced feature fusion through improved architectural components
    - Better preservation of fine-grained details from lower levels
    
    The PAN process:
    1. Downsamples P3 features by 2x using stride-2 convolution (256 channels)
    2. Concatenates downsampled P3 with processed P4 features from FPN
    3. Processes concatenated features through C2f block for feature fusion (512 channels)
    4. Downsamples processed P4 features by 2x using stride-2 convolution (512 channels)
    5. Concatenates downsampled P4 with original P5 features from backbone
    6. Processes final concatenated features through C2f block (512 channels)
    
    This bottom-up information flow creates stronger feature pyramids by shortening
    the information path between lower and higher pyramid levels, which is especially
    beneficial for detecting small objects that require fine-grained localization.
    
    Args:
        p3_out (keras.KerasTensor): Enhanced P3 features from FPN top-down pathway
            with shape corresponding to 256 * width_multiple channels at 1/8 scale.
        p4_processed (keras.KerasTensor): Enhanced P4 features from FPN top-down pathway
            with shape corresponding to 512 * width_multiple channels at 1/16 scale.
        p5_features (keras.KerasTensor): Original P5 features from backbone network
            with shape corresponding to 512 * width_multiple channels at 1/32 scale.
        width_multiple (float): Scaling factor for channel width. Controls the number
            of channels in each layer (e.g., 0.25 for YOLOv8n, 0.5 for YOLOv8s, 1.0 for YOLOv8m).
        depth_multiple (float): Scaling factor for layer depth. Controls the number
            of repeated blocks in C2f layers (e.g., 0.33 for YOLOv8n, 0.5 for YOLOv8s, 1.0 for YOLOv8m).
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
            - p5_out: Enhanced P5 features after PAN processing with 512 * width_multiple
              channels at 1/32 scale. Used for detecting large objects.
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
            images_input, width_multiple, depth_multiple, data_format
        )

        p3_out, p4_processed, p5_features = build_fpn(
            p3_features,
            p4_features,
            p5_features,
            width_multiple,
            depth_multiple,
            data_format,
        )

        feature_maps = build_pan(
            p3_out, p4_processed, p5_features, width_multiple, depth_multiple, data_format
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

@register_model
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

@register_model
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

@register_model
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

@register_model
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