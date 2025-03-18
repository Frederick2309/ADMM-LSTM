#! /bin/bash

if ! (command -v conda &> /dev/null); then
  echo "Anaconda not found. You must install anaconda first."
  exit 1
fi

if [ "$#" -lt 1 ]; then
  echo "Usage: bash install.sh <conda_env_name>. Missing conda environment name."
  exit 2
fi

echo "Trying to activate conda env: $2"

conda init
conda activate "$2"

python3 -m pip install --upgrade pip

if command -v nvidia-smi &> /dev/null
 then
  INSTALL_COMMAND="python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
 else
  INSTALL_COMMAND="python3 -m pip install  torch torchvision torchaudio"
fi

$INSTALL_COMMAND

python3 -m pip install -r requirements.txt

echo "Installation finished."
