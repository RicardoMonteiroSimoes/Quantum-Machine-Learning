#!/usr/bin/env bash
set -e
set -o pipefail

printf "Local Python paths\n"
printf 'python: %s\n' "$(which python)"
printf 'pip: %s\npip3: %s\n' "$(which pip)" "$(which pip3)"

printf "\n"

printf "Install/Upgrade Pennylane ...\n"
pip install pennylane --upgrade

printf "Install Pennylane Plugins ...\n"
pip install \
  pennylane-sf \
  pennylane-qiskit \
  amazon-braket-pennylane-plugin \
  pennylane-cirq \
  pennylane-forest \
  pennylane-qsharp

printf "Install Interfaces ...\n"
pip install autograd "tensorflow>=1.13.2" jax jaxlib
pip install torch torchvision torchaudio

printf "Install Amazon Braket Python SDK ...\n"
pip install amazon-braket-sdk

printf "Install/Upgrade boto3 ...\n"
pip install boto3 --upgrade
