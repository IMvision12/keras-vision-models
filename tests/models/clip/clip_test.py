from keras import ops

from kvmm.models import clip

from ...test_modelling import ModelTestCase


class TestCLIP(ModelTestCase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        super().setUp()

        batch_size = 2
        image_size = 224
        context_length = 77

        images = ops.ones((batch_size, image_size, image_size, 3))
        token_ids = ops.ones((batch_size, context_length), dtype="int32")
        padding_mask = ops.ones((batch_size, context_length), dtype="int32")

        self.input_data = {
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

        self.expected_output_shape = {
            "image_logits": (batch_size, batch_size),
            "text_logits": (batch_size, batch_size),
        }

        self.configure(
            model_cls=clip.ClipVitBase32,
            model_type="vlm",
            init_kwargs={
                "weights": None,
                "input_shape": (image_size, image_size, 3),
            },
            input_data=self.input_data,
            expected_output_shape=self.expected_output_shape,
        )

    def test_weight_initialization(self):
        custom_model = clip.ClipVitBase32(
            input_shape=(224, 224, 3),
            weights="openai_224",
        )
        return super().test_weight_initialization(custom_model)
