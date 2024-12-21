from kv.models.vision_transformer import ViTTiny16

model = ViTTiny16(input_shape=(180, 180, 3), weights=None)
model.load_weights("vit_tiny_patch16_384_augreg_in21k_ft_in1k.keras")
