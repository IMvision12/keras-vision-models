import pytest

from kvmm.models import segformer
from kvmm.tests.test_segmentation_modeling import (
    SegmentationModelConfig,
    SegmentationTest,
)


class TestSegFormer(SegmentationTest):
    @pytest.fixture
    def model_config(self) -> SegmentationModelConfig:
        return SegmentationModelConfig(
            model_cls=segformer.SegFormerB0, input_shape=(32, 32, 3)
        )

    def test_auxiliary_outputs(self, model_config):
        pytest.skip("SegFormer doesn't produce auxiliary outputs")
