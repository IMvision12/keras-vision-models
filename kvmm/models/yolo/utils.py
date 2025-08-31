from keras import ops


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