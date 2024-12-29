import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


def get_conversion_files():
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
    return f"convert_{param.parent.name}"


conversion_files = get_conversion_files()

pytestmark = pytest.mark.skipif(
    len(conversion_files) == 0, reason="No conversion files found in models directory"
)


def test_all_model_conversions(capsys):
    if not conversion_files:
        pytest.skip("No conversion files found in models directory")
    
    for conversion_file in conversion_files:
        start_time = time.time()

        assert conversion_file.exists(), f"Conversion file not found: {conversion_file}"
        assert conversion_file.is_file(), f"Not a file: {conversion_file}"

        try:
            with capsys.disabled():
                print(f"\nTesting conversion for {conversion_file.parent.name}...")

            env = os.environ.copy()
            env["TESTING"] = "1"

            with open(conversion_file, "r") as f:
                content = f.read()

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
                with capsys.disabled():
                    print(f"\n✗ Conversion failed for {conversion_file.parent.name}")
                    print(f"STDOUT:\n{result.stdout}")
                    print(f"STDERR:\n{result.stderr}")
                pytest.fail(f"Conversion failed for {conversion_file.parent.name} with return code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

            with capsys.disabled():
                print(
                    f"✓ Conversion successful for {conversion_file.parent.name} ({duration:.1f}s)"
                )

        except subprocess.TimeoutExpired:
            pytest.fail(f"Conversion timed out for {conversion_file.parent.name} after 300 seconds")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Process error for {conversion_file.parent.name}: {str(e)}\nSTDERR:\n{e.stderr}")
        except Exception as e:
            pytest.fail(f"Unexpected error for {conversion_file.parent.name}: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])