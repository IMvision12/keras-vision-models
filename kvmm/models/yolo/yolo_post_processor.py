import keras
from keras import ops
from yolo.layers import DFL
from yolo.utils import decode_bboxes, make_anchors


class YoloPostProcessor(keras.layers.Layer):
    """
    YOLO output processing layer that handles bbox decoding and class predictions.

    This layer processes raw YOLO outputs from multiple scales and converts them
    into decoded bounding boxes with class probabilities. It takes the raw feature
    maps from different detection heads and transforms them into final detections
    with proper coordinate decoding and class probability computation.

    The processing pipeline includes:
    1. Reshaping multi-scale feature maps into a unified format
    2. Separating bounding box and classification predictions
    3. Generating anchor points for each feature map scale
    4. Applying Distribution Focal Loss (DFL) for bbox regression
    5. Decoding bounding boxes from anchor-relative format to absolute coordinates
    6. Scaling boxes according to feature map strides
    7. Applying sigmoid activation to class predictions

    Args:
        strides (list of int, optional): Feature pyramid network strides for each
            detection scale. Represents the downsampling factor from input image
            to feature map. Defaults to [8, 16, 32] for small, medium, and large
            object detection respectively.
        num_classes (int, optional): Number of object classes to detect.
            Defaults to 80 (COCO dataset classes).
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Attributes:
        strides (list): Stride values for each detection scale
        num_classes (int): Number of detection classes
        dfl (DFL): Distribution Focal Loss layer for bbox regression with 16 bins

    Input Shape:
        List of 3D tensors with shape (batch_size, height, width, channels)
        where channels = 64 + num_classes (64 for bbox, rest for classes)
        Each tensor corresponds to a different detection scale.

    Output Shape:
        3D tensor with shape (batch_size, total_anchors, 4 + num_classes)
        where:
        - total_anchors is the sum of all anchor points across scales
        - First 4 channels contain decoded bounding box coordinates [x, y, w, h]
        - Remaining channels contain class probabilities (sigmoid activated)

    Example:
        ```python
        # Initialize post-processor for COCO detection
        post_processor = YoloPostProcessor(
            strides=[8, 16, 32],
            num_classes=80
        )

        # Process YOLO head outputs
        head_outputs = [small_scale_output, medium_scale_output, large_scale_output]
        detections = post_processor(head_outputs)

        # detections shape: (batch_size, total_anchors, 84)
        # where 84 = 4 (bbox) + 80 (classes)
        ```

    Note:
        - Assumes DFL (Distribution Focal Loss) with 16 bins is used for bbox regression
        - Input feature maps should have 144 channels (64 for bbox + 80 for classes)
        - Output bounding boxes are in absolute coordinates scaled by stride values
        - Class predictions are sigmoid-activated and ready for confidence thresholding
    """

    def __init__(self, strides=[8, 16, 32], num_classes=80, **kwargs):
        super().__init__(**kwargs)
        self.strides = strides
        self.num_classes = num_classes
        self.dfl = DFL(16)

    def build(self, input_shape):
        super().build(input_shape)
        dummy_input_shape = (None, 64, None)
        self.dfl.build(dummy_input_shape)

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

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "strides": self.strides,
                "num_classes": self.num_classes,
            }
        )
        return config
