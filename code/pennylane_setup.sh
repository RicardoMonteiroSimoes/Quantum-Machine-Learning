#!/usr/bin/env bash
set -e
set -o pipefail

printf "Local Python paths\n"
printf 'python: %s\n' "$(which python)"
printf 'pip: %s\npip3: %s\n' "$(which pip)" "$(which pip3)"

printf "\n"

printf "\nInstall/Upgrade Pennylane ...\n"
pip install pennylane --upgrade

printf "\nInstall Pennylane Plugins ...\n"
pip install \
  pennylane-sf \
  pennylane-qiskit \
  amazon-braket-pennylane-plugin \
  pennylane-cirq \
  pennylane-forest \
  pennylane-qsharp

printf "\nInstall Interfaces ...\n"
pip install autograd "tensorflow>=1.13.2" jax jaxlib
pip install torch torchvision torchaudio

printf "\nInstall Amazon Braket Python SDK ...\n"
pip install amazon-braket-sdk

printf "\nInstall/Upgrade boto3 ...\n"
pip install boto3 --upgrade

printf "\nInstall/Upgrade sklearn ...\n"
pip install sklearn --upgrade

printf "\Installing/Upgrading matplotlip draw library ...\n"
pip install pylatexenc --upgrade

printf "\Installing/Upgrading qiskit machine learning library ...\n"
pip install qiskit-machine-learning --upgrade
pip install qiskit-optimization --upgrade