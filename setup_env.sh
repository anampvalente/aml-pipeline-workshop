#!/bin/bash
 
set -e
 
sudo -u azureuser -i <<'EOF'
 
ENVIRONMENT_NAME=train_env_311
PYTHON_VERSION=3.11
KERNEL_DISPLAY_NAME='Train env (3.11)'
source /anaconda/etc/profile.d/conda.sh
echo "Creating conda environment"
conda create --name "$ENVIRONMENT_NAME" python="$PYTHON_VERSION" -y
echo "Activating environment '$ENVIRONMENT_NAME'..."
conda activate "$ENVIRONMENT_NAME"
conda install ipykernel -y
echo "Register environment with Jupyter kernel"
python -m ipykernel install --user --name "$ENVIRONMENT_NAME" --display-name "$KERNEL_DISPLAY_NAME"
conda deactivate
echo "You should close the terminal, refresh kernel list to see new environment you just created"
EOF