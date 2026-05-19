#!/bin/bash

set -eo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/environment.yml"
    exit 1
fi

ENV_FILE="$1"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: environment file not found:"
    echo "$ENV_FILE"
    exit 1
fi

echo "Loading modules"

ml load tools/prod
ml load miniforge/3

echo "Checking if miniforge is already installed..."

if [ ! -d "$HOME/miniforge3"]; then
    echo "no install found... Setting up miniforge"
    miniforge-setup
else 
    echo "miniforge already installed."
fi

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"

echo "Running Conda dry-run"
conda env create -f "$ENV_FILE" --solver=libmamba --dry-run

echo "Creating conda environment"
conda env create -f "$ENV_FILE" --solver=libmamba