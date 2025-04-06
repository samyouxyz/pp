#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse
import re


def create_virtualenv(venv_path):
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists(venv_path):
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            print(f"Created virtual environment at {venv_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create virtual environment: {e}")
            sys.exit(1)


def get_requirements_from_txt(venv_path):
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        pip_path = os.path.join(
            venv_path, "bin" if os.name != "nt" else "Scripts", "pip"
        )
        subprocess.run([pip_path, "install", "-r", req_file], check=True)


def install_dependencies(venv_path, requirements):
    """Install dependencies in the virtual environment."""
    if not requirements:
        print("No external dependencies detected")
        return

    pip_path = os.path.join(venv_path, "bin" if os.name != "nt" else "Scripts", "pip")
    try:
        # Update pip first
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        # Install all requirements
        for req in requirements:
            print(f"Installing {req}...")
            subprocess.run([pip_path, "install", req], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)


def run_python_file(venv_path, file_path):
    """Execute the Python file in the virtual environment."""
    python_path = os.path.join(
        venv_path, "bin" if os.name != "nt" else "Scripts", "python"
    )
    try:
        subprocess.run([python_path, file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run the Python file: {e}")
        sys.exit(1)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run a Python file in a new virtual environment with its dependencies"
    )
    parser.add_argument("file", help="Path to the Python file to execute")
    parser.add_argument(
        "--venv",
        default=".venv",
        help="Path to create virtual environment (default: .venv)",
    )

    args = parser.parse_args()

    # Ensure the file exists and is a Python file
    if not args.file.endswith(".py"):
        print("Error: Please provide a .py file")
        sys.exit(1)
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found")
        sys.exit(1)

    # Create virtual environment
    create_virtualenv(args.venv)

    # Get and install requirements
    requirements = get_requirements_from_txt(args.venv)
    install_dependencies(args.venv, requirements)

    # Run the Python file
    print(f"\nRunning {args.file} in virtual environment...")
    run_python_file(args.venv, args.file)


if __name__ == "__main__":
    main()
