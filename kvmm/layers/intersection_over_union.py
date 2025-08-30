import keras
from keras import ops


@keras.saving.register_keras_serializable(package="kvmm")
class IntersectionOverUnion(keras.layers.Layer):
    """Keras layer that calculates Intersection over Union (IoU) between bounding boxes.
    
    This layer computes the Intersection over Union (IoU) metric between two sets of 
    bounding boxes. IoU is a standard metric in object detection and computer vision 
    tasks that measures the overlap between predicted and ground truth bounding boxes.
    The layer can be used as a standalone component for IoU calculations or as part
    of other operations like Non-Maximum Suppression, object detection evaluation,
    or loss computation.

    The IoU is calculated as:
    IoU = Area of Intersection / Area of Union
    
    Where:
    - Area of Intersection: Overlapping area between two bounding boxes
    - Area of Union: Combined area of both boxes minus the intersection

    Arguments:
        **kwargs: Additional layer arguments passed to the parent Layer class.
            Standard Keras layer arguments like name, dtype, trainable are supported.

    Input shape:
        Two tensors representing bounding boxes in format [x1, y1, x2, y2]:
        - boxes1: Tensor of shape `(N, 4)` where N is the number of boxes in the first set
        - boxes2: Tensor of shape `(M, 4)` where M is the number of boxes in the second set
        
        Box coordinates should be:
        - x1, y1: Top-left corner coordinates
        - x2, y2: Bottom-right corner coordinates
        - Coordinates can be in any consistent unit (pixels, normalized, etc.)

    Output shape:
        2D tensor with shape `(N, M)` representing the IoU matrix where:
        - output[i, j] = IoU between boxes1[i] and boxes2[j]
        - Values range from 0.0 (no overlap) to 1.0 (perfect overlap)

    Example:
        ```python
        # Create IoU layer
        iou_layer = IntersectionOverUnion()
        
        # Example bounding boxes (x1, y1, x2, y2 format)
        boxes1 = tf.constant([[10, 10, 50, 50], [20, 20, 60, 60]])  # Shape: (2, 4)
        boxes2 = tf.constant([[15, 15, 45, 45], [100, 100, 120, 120]])  # Shape: (2, 4)
        
        # Calculate IoU matrix
        iou_matrix = iou_layer([boxes1, boxes2])  # Shape: (2, 2)
        
        # iou_matrix[0, 0] = IoU between first box in boxes1 and first box in boxes2
        # iou_matrix[0, 1] = IoU between first box in boxes1 and second box in boxes2
        # etc.
        ```

    Methods:
        calculate_iou(box1, box2): Calculate IoU between two individual bounding boxes.
            Args:
                box1: Tensor of shape (4,) representing first box [x1, y1, x2, y2]
                box2: Tensor of shape (4,) representing second box [x1, y1, x2, y2]
            Returns:
                Scalar tensor representing IoU value between 0.0 and 1.0

        calculate_iou_matrix(boxes1, boxes2): Calculate IoU matrix between two sets of boxes.
            Args:
                boxes1: Tensor of shape (N, 4) representing first set of boxes
                boxes2: Tensor of shape (M, 4) representing second set of boxes
            Returns:
                Tensor of shape (N, M) representing pairwise IoU values

    Notes:
        - A small epsilon (1e-7) is added to the union area to prevent division by zero
        - The layer handles cases where boxes have zero or negative width/height
        - Boxes with invalid coordinates (x2 <= x1 or y2 <= y1) will have zero area
        - The layer is differentiable and can be used in gradient-based optimization
        - Memory complexity is O(N*M) for N and M input boxes
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        boxes1, boxes2 = inputs
        return self.calculate_iou_matrix(boxes1, boxes2)
    
    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1[0], box1[1], box1[2], box1[3]
        x1_2, y1_2, x2_2, y2_2 = box2[0], box2[1], box2[2], box2[3]
        
        x1_inter = ops.maximum(x1_1, x1_2)
        y1_inter = ops.maximum(y1_1, y1_2)
        x2_inter = ops.minimum(x2_1, x2_2)
        y2_inter = ops.minimum(y2_1, y2_2)
        
        inter_width = ops.maximum(0.0, x2_inter - x1_inter)
        inter_height = ops.maximum(0.0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = area1 + area2 - inter_area
        iou = inter_area / (union_area + 1e-7)
        
        return iou
    
    def calculate_iou_matrix(self, boxes1, boxes2):
        boxes1_expanded = ops.expand_dims(boxes1, axis=1)
        boxes2_expanded = ops.expand_dims(boxes2, axis=0)
        
        x1_1 = boxes1_expanded[:, :, 0]
        y1_1 = boxes1_expanded[:, :, 1]
        x2_1 = boxes1_expanded[:, :, 2]
        y2_1 = boxes1_expanded[:, :, 3]
        
        x1_2 = boxes2_expanded[:, :, 0]
        y1_2 = boxes2_expanded[:, :, 1]
        x2_2 = boxes2_expanded[:, :, 2]
        y2_2 = boxes2_expanded[:, :, 3]
        
        x1_inter = ops.maximum(x1_1, x1_2)
        y1_inter = ops.maximum(y1_1, y1_2)
        x2_inter = ops.minimum(x2_1, x2_2)
        y2_inter = ops.minimum(y2_1, y2_2)
        
        inter_width = ops.maximum(0.0, x2_inter - x1_inter)
        inter_height = ops.maximum(0.0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = area1 + area2 - inter_area
        iou_matrix = inter_area / (union_area + 1e-7)
        
        return iou_matrix
    
    def compute_output_shape(self, input_shape):
        boxes1_shape, boxes2_shape = input_shape
        return (boxes1_shape[0], boxes2_shape[0])
    
    def get_config(self):
        config = super().get_config()
        return config