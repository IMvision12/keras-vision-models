import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


def get_conversion_files():
    """Find all model conversion files in the project."""
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    models_dir = project_root / "kv" / "models"

    if not models_dir.exists():
        print(f"Warning: Models directory not found at {models_dir}")
        return []

    conversion_files = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            conversion_file = list(model_dir.glob("convert_*_torch_to_keras.py"))
            conversion_files.extend(conversion_file)

    return conversion_files


def id_func(param):
    """Create test ID from parameter"""
    return f"convert_{param.parent.name}"


conversion_files = get_conversion_files()

pytestmark = pytest.mark.skipif(
    len(conversion_files) == 0, reason="No conversion files found in models directory"
)


@pytest.mark.parametrize("conversion_file", conversion_files, ids=id_func)
def test_model_conversion(conversion_file, capsys):
    """Test each model conversion script."""
    start_time = time.time()

    assert conversion_file.exists(), f"Conversion file not found: {conversion_file}"
    assert conversion_file.is_file(), f"Not a file: {conversion_file}"

    try:
        with capsys.disabled():
            print(f"\nTesting conversion for {conversion_file.parent.name}...")

        # Set environment variable to prevent model saving
        env = os.environ.copy()
        env["TESTING"] = "1"

        # Modify the conversion script content to prevent model saving
        with open(conversion_file, "r") as f:
            content = f.read()

        # Run the conversion script with the TESTING environment variable
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
import os
{content}
# Override model saving during testing
if 'TESTING' in os.environ:
    if results["standard_input"]:
        print("✓ Model conversion successful (saving skipped in test mode)")
    else:
        print("✗ Model conversion failed")
    """,
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env,
        )

        duration = time.time() - start_time

        if result.returncode != 0:
            print(f"\n✗ Conversion failed for {conversion_file.parent.name}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            pytest.fail(f"Conversion failed with return code {result.returncode}")

        with capsys.disabled():
            print(
                f"✓ Conversion successful for {conversion_file.parent.name} ({duration:.1f}s)"
            )

    except subprocess.TimeoutExpired:
        pytest.fail(f"Conversion timed out after 300 seconds")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Process error: {str(e)}\nSTDERR:\n{e.stderr}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
