"""
Model Equivalence Verification Utility for PyTorch and Keras Models

This module provides functionality to verify the equivalence of neural network models
between PyTorch and Keras frameworks, as well as between different Keras models.
It performs comprehensive testing of model outputs and optional performance benchmarking.

Key Features:
- Supports comparison between PyTorch and Keras models
- Supports comparison between different Keras models
- Validates model outputs across different batch sizes
- Performs optional performance benchmarking
- Provides detailed test results and diagnostics
- Handles both single and batch inference
- Supports custom tolerance levels for output comparison

Dependencies:
- numpy
- tensorflow
- torch
- keras
- typing
- time

Example Usage:
    # For PyTorch to Keras comparison
    results = verify_model_equivalence(
        model_a=torch_model,           # PyTorch model
        model_b=keras_model,           # Keras model
        input_shape=(224, 224, 3),     # Input shape without batch dimension
        output_specs={"num_classes": 1000},
        comparison_type="torch_to_keras",
        batch_sizes=[1, 4, 8],
        run_performance=True
    )

    # For Keras to Keras comparison
    results = verify_model_equivalence(
        model_a=keras_model_1,         # First Keras model
        model_b=keras_model_2,         # Second Keras model
        input_shape=(224, 224, 3),     # Input shape without batch dimension
        output_specs={"num_classes": 1000},
        comparison_type="keras_to_keras",
        batch_sizes=[1, 4, 8],
        run_performance=True
    )

Return Value:
    Returns a dictionary containing test results:
    {
        "standard_input": bool,              # Result of single sample test
        "batch_size_N": bool,                # Results for each batch size
        "performance": {                     # Optional performance metrics
            "model_a_inference_time": float,
            "model_b_inference_time": float,
            "time_ratio": float
        }
    }

Notes:
- Input shapes should be specified without the batch dimension
- The function handles necessary tensor transpositions for PyTorch inputs
- Performance testing runs multiple inferences to get average timing
- Different random seeds are used for reproducibility
- Custom tolerance levels can be set for numerical comparison
"""

from time import time
from typing import Any, Dict, List, Tuple, Union

import keras
import numpy as np
import tensorflow as tf
import torch


def verify_model_equivalence(
    model_a: Union[keras.Model, torch.nn.Module],
    model_b: keras.Model,
    input_shape: Union[Tuple[int, ...], List[int]],
    output_specs: Dict[str, Any],
    comparison_type: str = "torch_to_keras",
    batch_sizes: List[int] = [2, 4],
    run_performance: bool = True,
    num_performance_runs: int = 5,
    seed: int = 2025,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Dict[str, Any]:
    results = {}

    if comparison_type not in ["torch_to_keras", "keras_to_keras"]:
        raise ValueError(
            "comparison_type must be either 'torch_to_keras' or 'keras_to_keras'"
        )

    if comparison_type == "torch_to_keras" and not isinstance(model_a, torch.nn.Module):
        raise ValueError(
            "model_a must be a PyTorch model when comparison_type is 'torch_to_keras'"
        )
    elif comparison_type == "keras_to_keras" and not isinstance(model_a, keras.Model):
        raise ValueError(
            "model_a must be a Keras model when comparison_type is 'keras_to_keras'"
        )

    if "num_classes" not in output_specs:
        raise ValueError("output_specs must contain 'num_classes' key")

    def get_expected_output_shape(batch_size: int) -> Tuple[int, ...]:
        if comparison_type == "torch_to_keras":
            return (batch_size, output_specs["num_classes"])
        else:
            sample_input = np.zeros([1] + list(input_shape), dtype="float32")
            output_shape = model_a(sample_input).shape[1:]
            return (batch_size, *output_shape)

    def prepare_input(batch_size: int) -> Tuple[Any, Any]:
        if isinstance(input_shape, tuple):
            input_shape_list = list(input_shape)
        else:
            input_shape_list = input_shape

        if comparison_type == "torch_to_keras":
            keras_input = np.random.uniform(
                size=[batch_size, *input_shape_list]
            ).astype("float32")
            torch_input = torch.from_numpy(np.transpose(keras_input, [0, 3, 1, 2]))
            return keras_input, torch_input
        else:
            test_input = np.random.uniform(size=[batch_size] + input_shape_list).astype(
                "float32"
            )
            return test_input, test_input

    def get_model_output(
        model: Union[keras.Model, torch.nn.Module],
        input_data: Union[np.ndarray, torch.Tensor],
        expected_shape: Tuple[int, ...],
    ) -> np.ndarray:
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                output = model(input_data)
                output = output.detach().cpu().numpy()
        else:
            output = keras.ops.convert_to_numpy(model(input_data, training=False))

        assert output.shape == expected_shape, (
            f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
        )
        return output

    def test_outputs(output_a: np.ndarray, output_b: np.ndarray) -> bool:
        try:
            np.testing.assert_allclose(output_a, output_b, rtol=rtol, atol=atol)
            return True
        except AssertionError:
            return False

    def test_standard_input() -> bool:
        print("\n=== Testing Standard Input Shape ===")
        np.random.seed(seed)
        torch.manual_seed(seed) if comparison_type == "torch_to_keras" else None
        tf.random.set_seed(seed)
        keras.utils.set_random_seed(seed)

        input_a, input_b = prepare_input(batch_size=1)
        expected_shape = get_expected_output_shape(batch_size=1)

        try:
            output_a = get_model_output(
                model_a,
                input_b if comparison_type == "torch_to_keras" else input_a,
                expected_shape,
            )
            output_b = get_model_output(model_b, input_a, expected_shape)

            success = test_outputs(output_a, output_b)
            print(
                f"{'✓' if success else '✗'} Output {'matched' if success else 'mismatched'} "
                f"for input shape {input_shape}"
            )

            if not success:
                max_diff = np.max(np.abs(output_a - output_b))
                mean_diff = np.mean(np.abs(output_a - output_b))
                results["standard_input_diff"] = {
                    "max_difference": float(max_diff),
                    "mean_difference": float(mean_diff),
                }

            return success
        except Exception as e:
            print(f"✗ Test failed: {str(e)}")
            return False

    def test_batch_processing() -> Dict[str, bool]:
        print("\n=== Testing Different Batch Sizes ===")
        batch_results = {}

        for batch_size in batch_sizes:
            if batch_size == 1:
                continue

            print(f"\nTesting batch size: {batch_size}")
            np.random.seed(seed)
            torch.manual_seed(seed) if comparison_type == "torch_to_keras" else None
            tf.random.set_seed(seed)
            keras.utils.set_random_seed(seed)

            input_a, input_b = prepare_input(batch_size)
            expected_shape = get_expected_output_shape(batch_size)

            try:
                output_a = get_model_output(
                    model_a,
                    input_b if comparison_type == "torch_to_keras" else input_a,
                    expected_shape,
                )
                output_b = get_model_output(model_b, input_a, expected_shape)

                success = test_outputs(output_a, output_b)
                batch_results[f"batch_size_{batch_size}"] = success
                print(
                    f"{'✓' if success else '✗'} Output {'matched' if success else 'mismatched'}"
                )

                if not success:
                    max_diff = np.max(np.abs(output_a - output_b))
                    mean_diff = np.mean(np.abs(output_a - output_b))
                    batch_results[f"batch_size_{batch_size}_diff"] = {
                        "max_difference": float(max_diff),
                        "mean_difference": float(mean_diff),
                    }

            except Exception as e:
                batch_results[f"batch_size_{batch_size}"] = False
                print(f"✗ Test failed for batch size {batch_size}: {str(e)}")

        return batch_results

    results["standard_input"] = test_standard_input()

    if results["standard_input"]:
        results.update(test_batch_processing())
    else:
        print("Skipping batch processing tests due to standard input test failure")
        return results

    if run_performance:
        print("\n=== Testing Performance ===")
        input_a, input_b = prepare_input(batch_sizes[0])

        def run_inference(model, input_data, is_torch: bool = False):
            if is_torch:
                model.eval()
                with torch.no_grad():
                    _ = model(input_data)
                    times = []
                    for _ in range(num_performance_runs):
                        start = time()
                        _ = model(input_data)
                        times.append(time() - start)
            else:
                _ = model(input_data, training=False)
                times = []
                for _ in range(num_performance_runs):
                    start = time()
                    _ = model(input_data, training=False)
                    times.append(time() - start)

            return np.mean(times)

        time_a = run_inference(
            model_a,
            input_b if comparison_type == "torch_to_keras" else input_a,
            is_torch=(comparison_type == "torch_to_keras"),
        )
        time_b = run_inference(model_b, input_a, is_torch=False)

        results["performance"] = {
            "model_a_inference_time": time_a,
            "model_b_inference_time": time_b,
            "time_ratio": time_b / time_a,
        }

        print(f"Model A average inference time: {time_a:.4f}s")
        print(f"Model B average inference time: {time_b:.4f}s")
        print(f"Time ratio (B/A): {time_b / time_a:.2f}x")

    print("\n=== Test Summary ===")
    all_tests = [results["standard_input"]] + [
        v
        for k, v in results.items()
        if k.startswith("batch_size_") and isinstance(v, bool)
    ]
    all_passed = all(all_tests)
    print(
        f"Standard Input Test: {'Passed ✓' if results['standard_input'] else 'Failed ✗'}"
    )
    print(f"Batch Processing Tests: {'Passed ✓' if all_passed else 'Failed ✗'}")

    return results
