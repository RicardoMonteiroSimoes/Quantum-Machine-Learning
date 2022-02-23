## Setup

### Installing Pennylane
https://pennylane.ai/install.html

`pip install pennylane --upgrade`

<hr> 

### Installing plugins

This is needed to run the circuits on various quantum hardware solutions.

#### qiskit
`pip install pennylane-qiskit`
#### StrawberryFields
`pip install pennylane-sf`
#### Google Cirq
`pip install pennylane-cirq`
#### Rigetti forest
`pip install pennylane-forest`
#### Microsoft Q#
`pip install pennylane-qsharp`
#### All at once
`pip install pennylane-sf pennylane-qiskit pennylane-cirq pennylane-forest pennylane-qsharp`

<hr>

### Installing interfaces

These interfaces seamlessly integrate various ML libraries with Pennylane

#### NumPy/Autograd
`pip install autograd`
#### TensorFlow
`pip install "tensorflow>=1.13.2"`
#### JAX
`pip install jax jaxlib`
#### All at once
`pip install autograd "tensorflow>=1.13.2" jax jaxlib`

### PyTorch
`pip3 install torch torchvision torchaudio`

