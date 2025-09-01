import math

import keras
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from keras import ops

# COCO class names
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    Generate anchor points and stride tensors for object detection.

    This function creates anchor points (grid coordinates) for each feature map level
    in a multi-scale object detection network. Each anchor point represents the center
    of a grid cell where objects can be detected.

    Args:
        feats (list): List of feature map shapes, where each element contains
                     [batch_size, height, width, channels] or similar format.
                     Only height (feats[i][1]) and width (feats[i][2]) are used.
        strides (list): List of stride values corresponding to each feature map level.
                       Stride represents the downsampling factor from input image to
                       feature map (e.g., stride=32 means 32x32 input pixels -> 1 feature pixel).
        grid_cell_offset (float, optional): Offset for grid cell centers. Default is 0.5,
                                          which places anchors at the center of each grid cell.

    Returns:
        tuple: A tuple containing:
            - anchor_points (Tensor): Concatenated anchor points of shape (N, 2) where N is
                                    total number of anchor points across all feature maps.
                                    Each row contains [x, y] coordinates.
            - stride_tensor (Tensor): Concatenated stride values of shape (N, 1) where each
                                    value corresponds to the stride of its anchor point.

    Example:
        >>> feats = [(1, 20, 20, 256), (1, 10, 10, 512)]  # Two feature map levels
        >>> strides = [16, 32]
        >>> anchors, strides_tensor = make_anchors(feats, strides)
        >>> print(anchors.shape)  # (500, 2) - 400 points from 20x20 + 100 from 10x10
        >>> print(strides_tensor.shape)  # (500, 1)
    """
    anchor_points = []
    stride_tensor = []

    for i, stride in enumerate(strides):
        h, w = feats[i][1], feats[i][2]
        sx = ops.arange(w, dtype="float32") + grid_cell_offset
        sy = ops.arange(h, dtype="float32") + grid_cell_offset
        sy_grid, sx_grid = ops.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(
            ops.reshape(ops.stack([sx_grid, sy_grid], axis=-1), (-1, 2))
        )
        stride_tensor.append(ops.full((h * w, 1), stride, dtype="float32"))

    return ops.concatenate(anchor_points, axis=0), ops.concatenate(
        stride_tensor, axis=0
    )


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    Convert distance predictions to bounding boxes.

    This function converts distance-based predictions (left, top, right, bottom distances
    from anchor points) to bounding box coordinates. This is commonly used in modern
    object detection models like FCOS, YOLO, etc.

    Args:
        distance (Tensor): Distance predictions of shape (..., 4) where the last dimension
                          contains [left, top, right, bottom] distances from anchor points.
        anchor_points (Tensor): Anchor point coordinates of shape (..., 2) containing
                               [x, y] coordinates of anchor centers.
        xywh (bool, optional): If True, return bounding boxes in [x_center, y_center, width, height]
                              format. If False, return in [x1, y1, x2, y2] format. Default is True.
        dim (int, optional): Dimension along which to split and concatenate. Default is -1.

    Returns:
        Tensor: Bounding boxes in the specified format:
               - If xywh=True: shape (..., 4) with [x_center, y_center, width, height]
               - If xywh=False: shape (..., 4) with [x1, y1, x2, y2]

    Example:
        >>> distance = torch.tensor([[10, 5, 15, 8]])  # [left, top, right, bottom]
        >>> anchors = torch.tensor([[50, 50]])  # [x, y] anchor center
        >>> bbox_xywh = dist2bbox(distance, anchors, xywh=True)
        >>> print(bbox_xywh)  # [[52.5, 51.5, 25, 13]] - [x_center, y_center, width, height]
        >>> bbox_xyxy = dist2bbox(distance, anchors, xywh=False)
        >>> print(bbox_xyxy)  # [[40, 45, 65, 58]] - [x1, y1, x2, y2]
    """
    lt, rb = ops.split(distance, 2, axis=dim)
    anchor_points_t = ops.transpose(anchor_points, (0, 2, 1))
    x1y1 = anchor_points_t - lt
    x2y2 = anchor_points_t + rb

    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return ops.concatenate([c_xy, wh], axis=dim)
    return ops.concatenate([x1y1, x2y2], axis=dim)


def decode_bboxes(bboxes, anchors, xywh=True):
    """
    Decode bounding box predictions using anchor points.

    This is a convenience wrapper around dist2bbox specifically for decoding
    bounding box predictions in object detection models. It assumes the input
    bboxes are distance predictions that need to be converted to actual
    bounding box coordinates.

    Args:
        bboxes (Tensor): Distance-based bounding box predictions of shape (N, 4)
                        where each row contains [left, top, right, bottom] distances
                        from corresponding anchor points.
        anchors (Tensor): Anchor point coordinates of shape (N, 2) where each
                         row contains [x, y] coordinates of anchor centers.
        xywh (bool, optional): If True, return bounding boxes in
                              [x_center, y_center, width, height] format.
                              If False, return in [x1, y1, x2, y2] format.
                              Default is True.

    Returns:
        Tensor: Decoded bounding boxes of shape (N, 4) in the specified format.

    Example:
        >>> # Distance predictions for 2 detections
        >>> bbox_preds = torch.tensor([[5, 3, 7, 4], [10, 8, 12, 9]])
        >>> anchor_pts = torch.tensor([[25, 25], [50, 50]])
        >>> decoded = decode_bboxes(bbox_preds, anchor_pts, xywh=True)
        >>> print(decoded.shape)  # (2, 4)
        >>> print(decoded)  # Decoded bounding boxes in [x_center, y_center, width, height] format
    """
    return dist2bbox(bboxes, anchors, xywh=xywh, dim=1)


def make_divisible(x, divisor=8):
    """
    Make a number divisible by a given divisor.

    This function is commonly used in neural network architectures to ensure
    channel numbers are divisible by specific values (typically 8 or 16) for
    optimal hardware performance and memory alignment. Many accelerators and
    GPU architectures perform better when tensor dimensions are multiples of
    certain values.

    Args:
        x (float or int): The input number to make divisible.
        divisor (int, optional): The divisor to make x divisible by. Default is 8.
                                Common values are 8, 16, or 32 depending on the
                                target hardware architecture.

    Returns:
        int: The smallest integer greater than or equal to x that is divisible by divisor.

    Examples:
        >>> make_divisible(23, 8)
        24
        >>> make_divisible(32, 8)
        32
        >>> make_divisible(15.7, 16)
        16
        >>> make_divisible(100, 8)
        104

    Note:
        This function always rounds up to ensure the result is at least as large as
        the input value, which is important for maintaining model capacity when
        scaling channel dimensions.
    """
    return math.ceil(x / divisor) * divisor


def scale_channels(channels, width_multiple):
    """
    Scale the number of channels by a width multiplier.

    This function is used in model scaling strategies (like EfficientNet scaling)
    to adjust the width (number of channels) of neural network layers. The scaled
    value is made divisible by 8 to ensure optimal hardware performance.

    Args:
        channels (int): The base number of channels to scale.
        width_multiple (float): The scaling factor for channel width.
                               Values > 1.0 increase channels (wider model),
                               values < 1.0 decrease channels (narrower model).
                               Common values: 0.25, 0.5, 0.75, 1.0, 1.25, 1.5.

    Returns:
        int: The scaled number of channels, made divisible by 8.

    Examples:
        >>> scale_channels(64, 1.0)    # No scaling
        64
        >>> scale_channels(64, 1.5)    # 1.5x wider
        96
        >>> scale_channels(64, 0.5)    # 0.5x narrower
        32
        >>> scale_channels(48, 1.25)   # 1.25x wider, rounded to divisible by 8
        64

    Note:
        This function is typically used when creating different variants of a model
        architecture (e.g., YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x) where
        each variant has different channel widths but the same overall structure.
    """
    return make_divisible(channels * width_multiple)


def scale_depth(depth, depth_multiple):
    """
    Scale the depth (number of layers/repeats) by a depth multiplier.

    This function is used in model scaling strategies to adjust the depth
    of neural network components, such as the number of repeated blocks
    in a layer. The result is always at least 1 to ensure valid layer construction.

    Args:
        depth (int): The base depth (number of layer repeats) to scale.
        depth_multiple (float): The scaling factor for depth.
                               Values > 1.0 increase depth (deeper model),
                               values < 1.0 decrease depth (shallower model).
                               Common values: 0.33, 0.67, 1.0, 1.33, 1.67.

    Returns:
        int: The scaled depth, guaranteed to be at least 1.

    Examples:
        >>> scale_depth(3, 1.0)     # No scaling
        3
        >>> scale_depth(3, 1.33)    # 1.33x deeper
        4
        >>> scale_depth(6, 0.67)    # 0.67x shallower
        4
        >>> scale_depth(2, 0.33)    # Very shallow, but at least 1
        1
        >>> scale_depth(1, 0.1)     # Always at least 1
        1

    Note:
        The function uses rounding to convert float results to integers, and
        ensures the minimum value is 1. This is important because having 0
        layers would break the model architecture. This scaling is commonly
        used in compound scaling methods where model depth is scaled alongside
        width and resolution.
    """
    return max(round(depth * depth_multiple), 1)


def visualize_yolo_detections(
    images, detections, classes=COCO_CLASSES, title_prefix="Detection"
):
    """
    Visualize YOLO detections on single image or batch of images using pure Keras 3 ops.

    Args:
        images: Single image or batch of images (Keras tensor or numpy array)
               - Single: (H, W, 3), (3, H, W), or (H, W)
               - Batch: (B, H, W, 3), (B, 3, H, W)
        detections: Detection results from NMS (Keras tensor or numpy array)
                   - Single: (N, 6) array [x1, y1, x2, y2, conf, cls]
                   - Batch: (B, N, 6) tensor or list of (N, 6) arrays
        classes: Dictionary mapping class IDs to names (default: COCO_CLASSES)
        title_prefix: Prefix for plot titles

    # Example usage with Keras tensors:
    import keras

    # Single image (Keras tensor)
    img = keras.ops.random.uniform((640, 480, 3))
    dets = keras.ops.convert_to_tensor([[100, 100, 200, 200, 0.95, 0]])
    visualize_yolo_detections(img, dets)

    # Batch of images (Keras tensor)
    batch_imgs = keras.ops.random.uniform((4, 640, 480, 3))
    batch_dets = keras.ops.random.uniform((4, 5, 6))  # 4 images, max 5 detections each
    visualize_yolo_detections(batch_imgs, batch_dets)

    # Custom classes
    custom_classes = {0: 'my_object', 1: 'another_object'}
    visualize_yolo_detections(img, dets, classes=custom_classes)

    """

    def to_numpy(x):
        if hasattr(x, "numpy"):
            return x.numpy()
        elif keras.backend.is_keras_tensor(x):
            return keras.backend.convert_to_numpy(x)
        else:
            return np.array(x)

    def process_image_keras(img):
        if not keras.backend.is_keras_tensor(img):
            img = keras.ops.convert_to_tensor(img)

        img_shape = keras.ops.shape(img)

        if len(img_shape) == 3 and img_shape[0] == 3:
            img = keras.ops.transpose(img, [1, 2, 0])
        elif len(img_shape) == 2:
            img = keras.ops.expand_dims(img, axis=-1)
            img = keras.ops.repeat(img, 3, axis=-1)

        img_max = keras.ops.max(img)
        img = keras.ops.where(img_max > 1.0, img / 255.0, img)

        img = keras.ops.clip(img, 0.0, 1.0)

        return img

    def process_detections_keras(dets):
        if not keras.backend.is_keras_tensor(dets):
            dets = keras.ops.convert_to_tensor(dets)

        dets_shape = keras.ops.shape(dets)
        if len(dets_shape) == 1:
            dets = keras.ops.expand_dims(dets, axis=0)

        return dets

    def plot_single_detection(img_tensor, dets_tensor, ax, title):
        img_np = to_numpy(img_tensor)
        det_np = to_numpy(dets_tensor)

        ax.imshow(img_np)

        colors = [
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
            "pink",
            "cyan",
            "brown",
            "gray",
        ]

        if len(det_np) == 0 or (len(det_np.shape) == 2 and det_np.shape[0] == 0):
            ax.set_title(f"{title}: 0 objects found", fontsize=12)
            ax.axis("off")
            return

        if len(det_np.shape) == 1:
            det_np = det_np.reshape(1, -1)

        for i, det in enumerate(det_np):
            if len(det) >= 6:
                x1, y1, x2, y2, conf, cls = det[:6]

                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                conf, cls = float(conf), int(cls)

                width = x2 - x1
                height = y2 - y1
                color = colors[i % len(colors)]
                rect = patches.Rectangle(
                    (x1, y1),
                    width,
                    height,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)

                class_name = classes.get(cls, f"Class{cls}")
                label = f"{class_name}: {conf:.2f}"
                ax.text(
                    x1,
                    y1 - 5,
                    label,
                    color=color,
                    fontsize=10,
                    weight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, pad=2),
                )

        ax.set_title(f"{title}: {len(det_np)} objects found", fontsize=12)
        ax.axis("off")

    if not keras.backend.is_keras_tensor(images):
        images = keras.ops.convert_to_tensor(images)

    if isinstance(detections, list):
        if len(detections) == 1 and hasattr(detections[0], "numpy"):
            detections = detections[0]
        else:
            detections = [
                keras.ops.convert_to_tensor(det)
                if not keras.backend.is_keras_tensor(det)
                else det
                for det in detections
            ]
    elif not keras.backend.is_keras_tensor(detections):
        detections = keras.ops.convert_to_tensor(detections)

    img_shape = keras.ops.shape(images)
    is_batch = len(img_shape) == 4

    if is_batch:
        batch_size = img_shape[0]

        batch_images = []
        for i in range(to_numpy(batch_size)):
            img_slice = images[i]
            processed_img = process_image_keras(img_slice)
            batch_images.append(processed_img)

        if isinstance(detections, list):
            batch_detections = [process_detections_keras(det) for det in detections]
        else:
            det_shape = keras.ops.shape(detections)
            if len(det_shape) == 3:
                batch_detections = []
                for i in range(to_numpy(batch_size)):
                    det_slice = detections[i]
                    batch_detections.append(process_detections_keras(det_slice))
            else:
                processed_det = process_detections_keras(detections)
                batch_detections = [processed_det] * to_numpy(batch_size)

        n_images = to_numpy(batch_size)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for i in range(n_images):
            det = (
                batch_detections[i]
                if i < len(batch_detections)
                else keras.ops.zeros((0, 6))
            )
            title = f"{title_prefix} {i + 1}"
            plot_single_detection(batch_images[i], det, axes[i], title)

        for i in range(n_images, len(axes)):
            axes[i].axis("off")

    else:
        processed_img = process_image_keras(images)
        processed_det = process_detections_keras(detections)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        plot_single_detection(processed_img, processed_det, ax, title_prefix)

    plt.tight_layout()
    plt.show()
