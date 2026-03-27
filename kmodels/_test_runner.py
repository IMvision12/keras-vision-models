"""Cross-platform test runner for keras-models.

Usage:
    kmodels-test <command>

Commands:
    all                  Full test suite (torch, excludes slow/link tests)
    backend-torch        Backend tests on torch
    backend-jax          Backend tests on jax
    backend-tf           Backend tests on tensorflow
    backend-numpy        Backend tests on numpy
    serialization        Serialization roundtrip tests
    saving               Model save/load tests
    data-format          channels_first/last tests (torch)
    data-format-gpu      channels_first on TF GPU
    layers               Layer unit tests
    links                Link validation (slow, requires network)
    gpu                  All GPU-only tests
    help                 Show this message
"""

import os
import subprocess
import sys

PYTEST = [sys.executable, "-m", "pytest"]


def _run(backend, *pytest_args):
    """Run pytest with the given backend and arguments."""
    env = os.environ.copy()
    if backend:
        env["KERAS_BACKEND"] = backend
    cmd = PYTEST + list(pytest_args)
    print(f"\n{'=' * 60}")
    print(f"  KERAS_BACKEND={backend or '(default)'}  {' '.join(pytest_args)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, env=env)
    return result.returncode


COMMANDS = {}


def command(name):
    def decorator(fn):
        COMMANDS[name] = fn
        return fn

    return decorator


@command("all")
def test_all():
    return _run(
        "torch",
        "tests/",
        "-v",
        "--durations=20",
        "-m",
        "not slow and not link_validation and not gpu",
    )


@command("backend-torch")
def test_backend_torch():
    return _run("torch", "tests/integration/test_backend_compatibility.py", "-v")


@command("backend-jax")
def test_backend_jax():
    return _run("jax", "tests/integration/test_backend_compatibility.py", "-v")


@command("backend-tf")
def test_backend_tf():
    return _run("tensorflow", "tests/integration/test_backend_compatibility.py", "-v")


@command("backend-numpy")
def test_backend_numpy():
    return _run("numpy", "tests/integration/test_backend_compatibility.py", "-v")


@command("serialization")
def test_serialization():
    return _run("torch", "tests/integration/test_serialization.py", "-v")


@command("saving")
def test_saving():
    return _run("torch", "tests/integration/test_model_saving.py", "-v")


@command("data-format")
def test_data_format():
    return _run("torch", "tests/integration/test_data_formats.py", "-v")


@command("data-format-gpu")
def test_data_format_gpu():
    return _run(
        "tensorflow",
        "tests/integration/test_data_formats.py",
        "-v",
        "-k",
        "channels_first",
    )


@command("layers")
def test_layers():
    return _run("torch", "tests/layers/", "-v")


@command("links")
def test_links():
    return _run(
        None,
        "tests/integration/test_config_links.py",
        "-v",
        "-m",
        "link_validation",
    )


@command("gpu")
def test_gpu():
    rc1 = _run("torch", "tests/", "-v", "-m", "gpu")
    rc2 = _run(
        "tensorflow",
        "tests/integration/test_data_formats.py",
        "-v",
        "-k",
        "channels_first",
    )
    return rc1 or rc2


@command("help")
def show_help():
    print(__doc__)
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        show_help()
        if len(sys.argv) >= 2 and sys.argv[1] not in COMMANDS:
            print(f"\nError: Unknown command '{sys.argv[1]}'")
            return 1
        return 0

    return COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
