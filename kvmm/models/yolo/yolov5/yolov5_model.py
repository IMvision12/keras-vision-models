import keras
from keras import layers, ops
from yolo.blocks import conv_block, c3_block, sppf_block
from yolo.head import detect_head
from yolo.utils import scale_channels, scale_depth

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
        name="YOLOv5",
        **kwargs,
    ):
        if data_format not in ['channels_last', 'channels_first']:
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

        if isinstance(input_tensor, dict):
            images_input = input_tensor.get("images") or layers.Input(
                shape=image_input_shape, name="images"
            )
            bboxes_input = input_tensor.get("bboxes") or layers.Input(
                shape=[max_boxes, 5], name="bboxes"  # 5 = [x1, y1, x2, y2, label]
            )
        else:
            images_input = layers.Input(shape=image_input_shape, name="images")
            bboxes_input = layers.Input(shape=[max_boxes, 5], name="bboxes")  # 5 = [x1, y1, x2, y2, label]

        inputs = {
            "images": images_input,
            "bboxes": bboxes_input
        }

        x = conv_block(images_input, scale_channels(64, width_multiple), k=6, s=2,
                       data_format=data_format, name_prefix="model.model.0")

        x = conv_block(x, scale_channels(128, width_multiple), k=3, s=2,
                       data_format=data_format, name_prefix="model.model.1")

        x = c3_block(x, scale_channels(128, width_multiple), n=scale_depth(3, depth_multiple), shortcut=True,
                     data_format=data_format, name_prefix="model.model.2")

        x = conv_block(x, scale_channels(256, width_multiple), k=3, s=2,
                       data_format=data_format, name_prefix="model.model.3")

        x = c3_block(x, scale_channels(256, width_multiple), n=scale_depth(6, depth_multiple), shortcut=True,
                     data_format=data_format, name_prefix="model.model.4")
        p3_features = x

        x = conv_block(x, scale_channels(512, width_multiple), k=3, s=2,
                       data_format=data_format, name_prefix="model.model.5")

        x = c3_block(x, scale_channels(512, width_multiple), n=scale_depth(9, depth_multiple), shortcut=True,
                     data_format=data_format, name_prefix="model.model.6")
        p4_features = x

        x = conv_block(x, scale_channels(1024, width_multiple), k=3, s=2,
                       data_format=data_format, name_prefix="model.model.7")

        x = c3_block(x, scale_channels(1024, width_multiple), n=scale_depth(3, depth_multiple), shortcut=True,
                     data_format=data_format, name_prefix="model.model.8")

        x = sppf_block(x, scale_channels(1024, width_multiple), k=5,
                       data_format=data_format, name_prefix="model.model.9")
        p5_features = x

        # Feature Pyramid Network (FPN) - Top-down pathway
        p5_reduced = conv_block(p5_features, scale_channels(512, width_multiple), k=1, s=1,
                               data_format=data_format, name_prefix="model.model.10")

        # Upsample by factor of 2 (since p4 is 2x larger than p5)
        p5_upsampled = layers.UpSampling2D(size=2, data_format=data_format,
                                          interpolation='nearest', name="model_model_11_upsample")(p5_reduced)

        if data_format == "channels_last":
            concat_axis = -1
        else:
            concat_axis = 1

        p4_concat = layers.Concatenate(axis=concat_axis, name="model_model_12_concat")([p5_upsampled, p4_features])

        p4_processed = c3_block(p4_concat, scale_channels(512, width_multiple), n=scale_depth(3, depth_multiple), shortcut=False,
                               data_format=data_format, name_prefix="model.model.13")

        p4_reduced = conv_block(p4_processed, scale_channels(256, width_multiple), k=1, s=1,
                               data_format=data_format, name_prefix="model.model.14")

        # Upsample by factor of 2 (since p3 is 2x larger than p4)
        p4_upsampled = layers.UpSampling2D(size=2, data_format=data_format,
                                          interpolation='nearest', name="model_model_15_upsample")(p4_reduced)

        p3_concat = layers.Concatenate(axis=concat_axis, name="model_model_16_concat")([p4_upsampled, p3_features])

        p3_out = c3_block(p3_concat, scale_channels(256, width_multiple), n=scale_depth(3, depth_multiple), shortcut=False,
                         data_format=data_format, name_prefix="model.model.17")

        # Path Aggregation Network (PAN) - Bottom-up pathway
        p3_downsampled = conv_block(p3_out, scale_channels(256, width_multiple), k=3, s=2,
                                   data_format=data_format, name_prefix="model.model.18")

        p4_final_concat = layers.Concatenate(axis=concat_axis, name="model_model_19_concat")([p3_downsampled, p4_reduced])

        p4_out = c3_block(p4_final_concat, scale_channels(512, width_multiple), n=scale_depth(3, depth_multiple), shortcut=False,
                         data_format=data_format, name_prefix="model.model.20")

        p4_downsampled = conv_block(p4_out, scale_channels(512, width_multiple), k=3, s=2,
                                   data_format=data_format, name_prefix="model.model.21")

        p5_final_concat = layers.Concatenate(axis=concat_axis, name="model_model_22_concat")([p4_downsampled, p5_reduced])

        p5_out = c3_block(p5_final_concat, scale_channels(1024, width_multiple), n=scale_depth(3, depth_multiple), shortcut=False,
                         data_format=data_format, name_prefix="model.model.23")

        feature_maps = [p3_out, p4_out, p5_out]
        detection_outputs = detect_head(feature_maps, nc=nc, reg_max=16,
                                      data_format=data_format, name_prefix="model.model.24")
        
        boxes = bboxes_input[:, :, :4]  # Extract [x1, y1, x2, y2]
        labels = bboxes_input[:, :, 4]  # Extract label

        # Create valid mask: valid if label >= 0 (assuming -1 for padding)
        valid_mask = ops.cast(labels >= 0, "float32")

        outputs = {
            "predictions": detection_outputs,
            "targets": {
                "boxes": boxes,
                "labels": labels,
                "valid_mask": valid_mask
            }
        }

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape[:1],
            "nc": self.nc,
            "data_format": self.data_format,
            "depth_multiple": self.depth_multiple,
            "width_multiple": self.width_multiple,
            "input_tensor": self.input_tensor,
            "name": self.name,
            "trainable": self.trainable,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)