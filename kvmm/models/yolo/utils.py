from keras import ops
import math

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
        sy_grid, sx_grid = ops.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(ops.reshape(ops.stack([sx_grid, sy_grid], axis=-1), (-1, 2)))
        stride_tensor.append(ops.full((h * w, 1), stride, dtype="float32"))

    return ops.concatenate(anchor_points, axis=0), ops.concatenate(stride_tensor, axis=0)


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