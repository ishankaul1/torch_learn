#!/bin/bash

# setup.sh
# ====================================
# Run this once to create the virtual environment
# and install dependencies

echo "Creating virtual environment 'torch_learn_env'..."
python3 -m venv torch_learn_env

echo "Activating virtual environment..."
source torch_learn_env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo "Run 'source start_venv.sh' to activate the environment in the future."