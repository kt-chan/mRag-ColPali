#!/bin/bash

# Check if Conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not available in PATH."
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "Error: requirements.txt not found in current directory."
    exit 1
fi

# Install or upgrade the uv tool
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
else
    echo "Updating uv..."
    pip install --upgrade uv
fi

# Create or check Conda environment
venv_path="./venv"
if [ -d "$venv_path" ]; then
    echo "Virtual environment exists at $venv_path. Checking if Python version is up to date..."
    conda update --prefix "$venv_path" --yes python=3.12
else
    echo "Creating conda environment at $venv_path..."
    conda create --prefix "$venv_path" --yes python=3.12
fi

# Get the Python executable path
python_path="$venv_path/python"
if [ ! -f "$python_path" ]; then
    python_path="$venv_path/bin/python"
fi


# Update UV cache directory
export UV_CACHE_DIR="$HOME/.cache/uv"

# Install or upgrade all packages using uv with the specified Python
echo "Installing or upgrading packages with uv..."
uv pip install --python "$python_path" --upgrade -r requirements.txt

echo "Setup completed successfully."