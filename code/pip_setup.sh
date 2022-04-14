#!/usr/bin/env bash
set -e
set -o pipefail

printf "Local Python paths\n"
printf 'python: %s\n' "$(which python)"
printf 'python3: %s\n' "$(which python3)"
printf 'pip: %s\npip3: %s\n' "$(which pip)" "$(which pip3)"

printf "\n"

printf "\nInstall/Upgrade Pennylane ...\n"
pip3 install pennylane --upgrade

printf "\nInstall Pennylane Plugins ...\n"
pip3 install \
  pennylane-sf \
  pennylane-qiskit \
  amazon-braket-pennylane-plugin \
  pennylane-cirq \
  pennylane-forest \
  pennylane-qsharp --upgrade

printf "\nInstall Interfaces ...\n"
pip3 install autograd "tensorflow>=1.13.2" jax jaxlib
pip3 install torch torchvision torchaudio

printf "\nInstall Amazon Braket Python SDK ...\n"
pip3 install amazon-braket-sdk --upgrade

printf "\nInstall/Upgrade boto3 ...\n"
pip3 install boto3 --upgrade

printf "\nInstall/Upgrade sklearn ...\n"
pip3 install sklearn --upgrade

printf "\Installing/Upgrading matplotlip draw library ...\n"
pip3 install pylatexenc --upgrade

printf "\Installing/Upgrading qiskit machine learning library ...\n"
pip3 install qiskit-machine-learning --upgrade
pip3 install qiskit-optimization --upgrade
pip3 install qiskit-aer-gpu --upgrade

printf "\Installing/Upgrading utilities ...\n"
pip3 install tqdm --upgrade
pip3 install GPUtil --upgrade

printf "\Installing/Upgrading pydot ...\n"
pip3 install pydot --upgrade

printf "\Adding additional libraries for notebook tooling ...\n"
pip3 install git+https://github.com/qiskit-community/qiskit-textbook.git#subdirectory=qiskit-textbook-src
pip3 install numexpr --upgrade

