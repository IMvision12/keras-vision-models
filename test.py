from kv.models import MiT_B0

model = MiT_B0(
    include_top=False,
    as_backbone=True,
    input_shape=(224, 224, 3),
    weights=None,
    include_preprocessing=False,
)
print(model.input_shape)
