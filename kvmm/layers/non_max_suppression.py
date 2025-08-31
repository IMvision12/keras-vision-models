import keras
from keras import ops

from kvmm.layers import NonMaxSuppression
from kvmm.models.yolo.layers import DFL
from kvmm.models.yolo.utils import decode_bboxes, make_anchors


class YoloPostProcessor(keras.layers.Layer):
    """
    YOLO output processing layer that handles bbox decoding, class predictions, and NMS.

    This layer processes raw YOLO outputs from multiple scales and converts them
    into final filtered detections through bbox decoding, class probability computation,
    and Non-Maximum Suppression. It takes the raw feature maps from different detection
    heads and transforms them into final detections ready for downstream tasks.

    The processing pipeline includes:
    1. Reshaping multi-scale feature maps into a unified format
    2. Separating bounding box and classification predictions
    3. Generating anchor points for each feature map scale
    4. Applying Distribution Focal Loss (DFL) for bbox regression
    5. Decoding bounding boxes from anchor-relative format to absolute coordinates
    6. Scaling boxes according to feature map strides
    7. Applying sigmoid activation to class predictions
    8. Performing Non-Maximum Suppression to filter overlapping detections

    Args:
        strides (list of int, optional): Feature pyramid network strides for each
            detection scale. Represents the downsampling factor from input image
            to feature map. Defaults to [8, 16, 32] for small, medium, and large
            object detection respectively.
        num_classes (int, optional): Number of object classes to detect.
            Defaults to 80 (COCO dataset classes).
        apply_nms (bool, optional): Whether to apply Non-Maximum Suppression.
            When True, returns filtered detections ready for use. When False,
            returns raw predictions suitable for custom post-processing.
            Defaults to True.
        conf_threshold (float, optional): Minimum confidence score for detections
            to be considered during NMS. Only used when apply_nms=True.
            Defaults to 0.25.
        iou_threshold (float, optional): IoU threshold for NMS suppression.
            Detections with IoU greater than this threshold are suppressed.
            Only used when apply_nms=True. Defaults to 0.7.
        max_detections (int, optional): Maximum number of detections to keep
            per image after NMS. Only used when apply_nms=True. Defaults to 300.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Attributes:
        strides (list): Stride values for each detection scale
        num_classes (int): Number of detection classes
        apply_nms (bool): Whether NMS is applied
        dfl (DFL): Distribution Focal Loss layer for bbox regression with 16 bins
        nms (NonMaxSuppression): NMS layer (only created when apply_nms=True)

    Input Shape:
        List of 3D tensors with shape (batch_size, height, width, channels)
        where channels = 64 + num_classes (64 for bbox, rest for classes)
        Each tensor corresponds to a different detection scale.

    Output Shape:
        When apply_nms=True:
            List of 2D tensors, one per batch item, each with shape (num_detections, 6)
            where each detection contains: [x1, y1, x2, y2, confidence, class_id]

        When apply_nms=False:
            3D tensor with shape (batch_size, total_anchors, 4 + num_classes)
            where:
            - total_anchors is the sum of all anchor points across scales
            - First 4 channels contain decoded bounding box coordinates [x, y, w, h]
            - Remaining channels contain class probabilities (sigmoid activated)

    Example:
        ```python
        # Initialize post-processor with NMS for inference
        post_processor = YoloPostProcessor(
            strides=[8, 16, 32],
            num_classes=80,
            apply_nms=True,
            conf_threshold=0.5,
            iou_threshold=0.6,
            max_detections=100
        )

        # Process YOLO head outputs
        head_outputs = [small_scale_output, medium_scale_output, large_scale_output]
        detections = post_processor(head_outputs)

        # detections is a list of tensors, one per batch item
        # Each tensor shape: (num_kept_detections, 6)
        # Format: [x1, y1, x2, y2, confidence, class_id]

        # For training (without NMS)
        training_processor = YoloPostProcessor(
            strides=[8, 16, 32],
            num_classes=80,
            apply_nms=False
        )
        raw_predictions = training_processor(head_outputs)
        # raw_predictions shape: (batch_size, total_anchors, 84)
        ```

    Note:
        - Assumes DFL (Distribution Focal Loss) with 16 bins is used for bbox regression
        - Input feature maps should have 144 channels (64 for bbox + 80 for classes)
        - When apply_nms=True, output bounding boxes are in XYXY format (x1, y1, x2, y2)
        - When apply_nms=False, output bounding boxes are in XYWH format (center_x, center_y, width, height)
        - Class predictions are sigmoid-activated in both cases
        - NMS converts coordinates from XYWH to XYXY format internally
    """

    def __init__(
        self,
        strides=[8, 16, 32],
        num_classes=80,
        apply_nms=True,
        conf_threshold=0.25,
        iou_threshold=0.7,
        max_detections=300,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strides = strides
        self.num_classes = num_classes
        self.apply_nms = apply_nms
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        self.dfl = DFL(16)

        if self.apply_nms:
            self.nms = NonMaxSuppression(
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                num_classes=num_classes,
            )

    def build(self, input_shape):
        super().build(input_shape)
        dummy_input_shape = (None, 64, None)
        self.dfl.build(dummy_input_shape)

        if self.apply_nms:
            dummy_nms_shape = (None, None, 4 + self.num_classes)
            self.nms.build(dummy_nms_shape)

    def call(self, inputs):
        x = inputs

        shapes = []
        for xi in x:
            shape = ops.shape(xi)
            shapes.append((shape[0], shape[1], shape[2]))

        x_reshaped = []
        for i, xi in enumerate(x):
            batch_size = shapes[i][0]
            h_w = shapes[i][1] * shapes[i][2]
            reshaped = ops.reshape(xi, (batch_size, h_w, 144))
            x_reshaped.append(reshaped)

        x_cat = ops.concatenate(x_reshaped, axis=1)

        box = x_cat[:, :, :64]
        cls = x_cat[:, :, 64:]

        anchors, stride_tensor = make_anchors(shapes, self.strides)

        batch_size = ops.shape(x_cat)[0]
        anchors_reshaped = ops.expand_dims(anchors, 0)
        anchors_reshaped = ops.repeat(anchors_reshaped, batch_size, axis=0)

        box_transposed = ops.transpose(box, (0, 2, 1))

        dfl_output = self.dfl(box_transposed)

        dbox = decode_bboxes(dfl_output, anchors_reshaped)

        stride_tensor_reshaped = ops.expand_dims(stride_tensor, 0)
        stride_tensor_reshaped = ops.repeat(stride_tensor_reshaped, batch_size, axis=0)
        stride_tensor_transposed = ops.transpose(stride_tensor_reshaped, (0, 2, 1))
        dbox = dbox * stride_tensor_transposed
        cls_sigmoid = ops.sigmoid(cls)
        dbox_transposed = ops.transpose(dbox, (0, 2, 1))
        output = ops.concatenate([dbox_transposed, cls_sigmoid], axis=2)

        if self.apply_nms:
            output = self.nms(output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "strides": self.strides,
                "num_classes": self.num_classes,
                "apply_nms": self.apply_nms,
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "max_detections": self.max_detections,
            }
        )
        return config
