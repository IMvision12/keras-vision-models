import pytest

from kvmm.models import segformer

from ....test_segmentation_modeling import SegmentationTestCase


class TestSegFormer(SegmentationTestCase):
    """Test case for the SegFormer model."""

    __test__ = True

    def setUp(self):
        super().setUp()
        self.configure(model_cls=segformer.SegFormerB0, input_shape=(32, 32, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "weights": None,
        }

    def test_weight_loading(self):
        custom_model = segformer.SegFormerB0(
            input_shape=(32, 32, 3),
            weights="ade20k_512",
        )
        return super().test_weight_loading(custom_model)
    
    def test_auxiliary_outputs(self):
        pytest.skip("SegFormer doesn't produce auxiliary outputs")