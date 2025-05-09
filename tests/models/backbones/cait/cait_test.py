from keras import ops

from kvmm.models import cait

from ....test_modelling import ModelTestCase


class TestCaiT(ModelTestCase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        super().setUp()

        self.input_data = ops.ones((2, 32, 32, 3))
        self.expected_output_shape = (2, 1000)

        self.configure(
            model_cls=cait.CaiTXXS24,
            model_type="backbone",
            init_kwargs={
                "weights": None,
                "input_shape": (32, 32, 3),
                "include_top": True,
            },
            input_data=self.input_data,
            expected_output_shape=self.expected_output_shape,
        )

    def test_weight_initialization(self):
        custom_model = cait.CaiTXXS24(
            input_shape=(32, 32, 3),
        )
        return super().test_weight_initialization(custom_model)

    def test_backbone_features(self):
        model = self.create_model(include_top=False, as_backbone=True)
        input_data = self.get_input_data()
        features = model(input_data)

        self.assertIsInstance(
            features, list, "Backbone output should be a list of feature maps"
        )

        self.assertGreaterEqual(
            len(features), 2, "Backbone should output at least 2 feature maps"
        )

        for i, feature_map in enumerate(features):
            is_transformer_output = len(feature_map.shape) == 3

            self.assertIn(
                len(feature_map.shape),
                (3, 4),
                f"Feature map {i} should be a 3D (transformer) or 4D (CNN) tensor, "
                f"got shape {feature_map.shape}",
            )

            self.assertEqual(
                feature_map.shape[0],
                self.batch_size,
                f"Feature map {i} has incorrect batch size. "
                f"Expected {self.batch_size}, got {feature_map.shape[0]}",
            )

            if is_transformer_output:
                seq_len, channels = feature_map.shape[1:]
                self.assertTrue(
                    seq_len > 0 and channels > 0,
                    f"Feature map {i} has invalid dimensions: "
                    f"sequence_length={seq_len}, channels={channels}",
                )

                if i > 0:
                    prev_map = features[i - 1]
                    prev_seq_len = prev_map.shape[1]

                    # Special case for CaiT models:
                    # The last feature map in CaiT may have the class token added
                    # which increases sequence length from 196 to 197
                    if i == len(features) - 1 and seq_len == prev_seq_len + 1:
                        # This is expected for CaiT's final feature map with class token
                        continue

                    self.assertLessEqual(
                        seq_len,
                        prev_seq_len,
                        f"Feature map {i} has larger sequence length than previous feature map. "
                        f"Got {seq_len}, previous was {prev_seq_len}",
                    )
