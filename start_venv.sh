#!/bin/bash

# INSTRUCTIONS
# ====================================
# Run source start_venv.sh to activate
# 'deactivate' to exit


source torch_learn_env/bin/activate

# Set PYTHONPATH to project root
export PYTHONPATH=$(pwd):$PYTHONPATH

# Optional: show that it's set
echo "PYTHONPATH set to: $PYTHONPATH"

