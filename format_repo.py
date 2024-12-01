import argparse
import os
import subprocess
from pathlib import Path


# run python format_repo.py .
def format_repository(repo_path: str, line_length: int = 88):
    """
    Format all Python files in a repository using Ruff and Isort.

    Args:
        repo_path: Path to the repository
        line_length: Maximum line length for formatting (default: 88)
    """
    repo_path = os.path.abspath(repo_path)
    print(f"Formatting repository at: {repo_path}")

    python_files = list(Path(repo_path).rglob("*.py"))

    python_files = [
        f for f in python_files if "venv" not in str(f) and ".env" not in str(f)
    ]

    if not python_files:
        print("No Python files found in the repository.")
        return

    print(f"Found {len(python_files)} Python files to format.")

    # Format with Ruff
    print("\nRunning Ruff formatting...")
    ruff_config = [
        "ruff",
        "format",
        "--line-length",
        str(line_length),
        *[str(f) for f in python_files],
    ]
    subprocess.run(ruff_config, check=True)

    # Sort imports with Isort
    print("\nRunning Isort for import sorting...")
    isort_config = [
        "isort",
        "--line-length",
        str(line_length),
        *[str(f) for f in python_files],
    ]
    subprocess.run(isort_config, check=True)

    print("\nFormatting complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format Python files in a repository using Ruff and Isort"
    )
    parser.add_argument("repo_path", help="Path to the repository")
    parser.add_argument(
        "--line-length", type=int, default=88, help="Maximum line length (default: 88)"
    )

    args = parser.parse_args()
    format_repository(args.repo_path, args.line_length)
