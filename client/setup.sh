#!/usr/bin/bash

command_exists () {
    type "$1" &> /dev/null ;
}

set -e
trap 'echo Error: Command failed!' ERR


if ! command_exists pipx; then
    echo "Pipx is not installed, Installing Pipx..."
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath
    source ~/.bashrc
fi

if ! command_exists poetry; then
    echo "Poetry is not installed, Installing Poetry..."
    pipx install poetry
fi

if [ ! -f "pyproject.toml" ]; then
    echo "No pyproject.toml file found, creating one..."
    poetry init
    poetry shell

    echo "The shell created by Poetry should be your active interpreter."
fi

if [ -f "requirements.txt" ]; then
    echo "Requirements.txt file found, installing dependencies..."
    poetry add $(cat requirements.txt)
else
    echo "Requirements.txt file not found, skipping dep installation..."
fi

echo "Setup complete!"
