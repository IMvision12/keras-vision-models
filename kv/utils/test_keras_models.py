from time import time
from typing import Any, Dict, List, Tuple, Union

import keras
import numpy as np
import tensorflow as tf
import torch


class ModelEquivalenceTester:
    def __init__(
        self,
        keras_model: keras.Model,
        torch_model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        output_specs: Dict[str, Any],
        rtol: float = 1e-4,
        atol: float = 1e-4,
        random_seed: int = 2023,
    ):
        self.keras_model = keras_model
        self.torch_model = torch_model
        self.input_shape = input_shape
        self.output_specs = output_specs
        self.rtol = rtol
        self.atol = atol
        self.random_seed = random_seed

        self._validate_output_specs()

    def _validate_output_specs(self):
        if "num_classes" not in self.output_specs:
            raise ValueError("Missing required output specification: num_classes")

    def _get_expected_output_shape(self, batch_size: int) -> Tuple[int, ...]:
        return (batch_size, self.output_specs["num_classes"])

    def _prepare_input(self, batch_size: int) -> Tuple[np.ndarray, torch.Tensor]:
        keras_input = np.random.uniform(size=[batch_size, *self.input_shape]).astype(
            "float32"
        )
        torch_input = torch.from_numpy(np.transpose(keras_input, [0, 3, 1, 2]))
        return keras_input, torch_input

    def _convert_output_to_numpy(
        self, output: Union[torch.Tensor, np.ndarray, tf.Tensor]
    ) -> np.ndarray:
        if isinstance(output, torch.Tensor):
            return output.detach().cpu().numpy()
        elif isinstance(output, tf.Tensor):
            return output.numpy()
        elif isinstance(output, np.ndarray):
            return output
        else:
            raise ValueError(f"Unsupported output type: {type(output)}")

    def test_model_output(
        self,
        model: Union[keras.Model, torch.nn.Module],
        input_data: Union[np.ndarray, torch.Tensor],
        expected_shape: Tuple[int, ...],
    ) -> np.ndarray:
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                output = model(input_data)
        else:
            output = model(input_data, training=False)

        output = self._convert_output_to_numpy(output)
        assert (
            output.shape == expected_shape
        ), f"Output shape mismatch: expected {expected_shape}, got {output.shape}"

        return output

    def test_outputs(self, keras_output: np.ndarray, torch_output: np.ndarray) -> bool:
        try:
            np.testing.assert_allclose(
                keras_output, torch_output, rtol=self.rtol, atol=self.atol
            )
            return True
        except AssertionError:
            return False

    def test_standard_input(self) -> bool:
        print("\n=== Testing Standard Input Shape ===")
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        keras_input, torch_input = self._prepare_input(batch_size=1)
        expected_shape = self._get_expected_output_shape(batch_size=1)

        try:
            torch_output = self.test_model_output(
                self.torch_model, torch_input, expected_shape
            )
            keras_output = self.test_model_output(
                self.keras_model, keras_input, expected_shape
            )

            success = self.test_outputs(keras_output, torch_output)
            print(
                f"{'✓' if success else '✗'} Output {'matched' if success else 'mismatched'} "
                f"for input shape {self.input_shape}"
            )
            return success
        except Exception as e:
            print(f"✗ Test failed: {str(e)}")
            return False

    def test_batch_processing(self, batch_sizes: list = [1, 4]) -> Dict[str, bool]:
        print("\n=== Testing Different Batch Sizes ===")
        results = {}

        for batch_size in batch_sizes:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            tf.random.set_seed(self.random_seed)

            keras_input, torch_input = self._prepare_input(batch_size)
            expected_shape = self._get_expected_output_shape(batch_size)

            try:
                torch_output = self.test_model_output(
                    self.torch_model, torch_input, expected_shape
                )
                keras_output = self.test_model_output(
                    self.keras_model, keras_input, expected_shape
                )

                success = self.test_outputs(keras_output, torch_output)
                results[f"batch_size_{batch_size}"] = success
                print(
                    f"{'✓' if success else '✗'} Output {'matched' if success else 'mismatched'} "
                    f"for batch size {batch_size}"
                )
            except Exception as e:
                results[f"batch_size_{batch_size}"] = False
                print(f"✗ Failed for batch size {batch_size}: {str(e)}")

        return results

    def test_performance(
        self, batch_size: int = 1, num_runs: int = 10
    ) -> Dict[str, float]:
        print("\n=== Testing Performance ===")
        keras_input, torch_input = self._prepare_input(batch_size)

        def run_inference(model_func, input_data):
            _ = model_func(input_data)  # warmup

            times = []
            for _ in range(num_runs):
                start = time()
                _ = model_func(input_data)
                times.append(time() - start)
            return np.mean(times)

        self.torch_model.eval()
        with torch.no_grad():
            torch_time = run_inference(lambda x: self.torch_model(x), torch_input)

        keras_time = run_inference(
            lambda x: self.keras_model(x, training=False), keras_input
        )

        results = {
            "torch_inference_time": torch_time,
            "keras_inference_time": keras_time,
        }

        print(f"PyTorch average inference time: {torch_time:.4f}s")
        print(f"Keras average inference time: {keras_time:.4f}s")

        return results


def run_all_tests(
    keras_model: keras.Model,
    torch_model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    output_specs: Dict[str, Any],
    random_seed: int = 2023,
    batch_sizes: List[int] = [1, 4],
    run_performance: bool = True,
) -> Dict[str, Any]:
    tester = ModelEquivalenceTester(
        keras_model=keras_model,
        torch_model=torch_model,
        input_shape=input_shape,
        output_specs=output_specs,
        random_seed=random_seed,
    )

    standard_result = tester.test_standard_input()
    batch_results = tester.test_batch_processing(batch_sizes)

    results = {
        "standard_input": standard_result,
        "batch_processing": batch_results,
    }

    if run_performance:
        performance_results = tester.test_performance()
        results["performance"] = performance_results

    print("\n=== Test Summary ===")
    print(f"Standard Input Test: {'Passed ✓' if standard_result else 'Failed ✗'}")
    print(
        f"Batch Processing Tests: {'Passed ✓' if all(batch_results.values()) else 'Failed ✗'}"
    )

    if run_performance:
        print(
            f"Performance Ratio (Keras/PyTorch): {performance_results['keras_inference_time'] / performance_results['torch_inference_time']:.2f}x"
        )

    return results
