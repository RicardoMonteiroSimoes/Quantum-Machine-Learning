#!/bin/bash
echo "Setting up Pennylane"
pip install pennylane --upgrade
pip install pennylane-sf pennylane-qiskit pennylane-cirq pennylane-forest pennylane-qsharp
pip install autograd "tensorflow>=1.13.2" jax jaxlib
pip3 install torch torchvision torchaudio