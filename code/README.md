- [Setup](#setup)
  - [Installing/Updating Pennylane](#installingupdating-pennylane)
  - [Installing plugins](#installing-plugins)
  - [Installing interfaces](#installing-interfaces)
  - [Installing other tools](#installing-other-tools)
# Setup

## Installing/Updating Pennylane
https://pennylane.ai/install.html

*[Anaconda](https://www.anaconda.com/products/individual) is recommended*

`pip install pennylane --upgrade`

<hr>

## Installing plugins

This is needed to run the circuits on various quantum hardware solutions.
More can be found here: https://pennylane.ai/plugins.html

| Pugin name                        | Install                                      |
| :-------------------------------- | :------------------------------------------- |
| qiskit                            | `pip install pennylane-qiskit`               |
| [Amazon Braket](Amazon_Braket.md) | `pip install amazon-braket-pennylane-plugin` |
| StrawberryFields                  | `pip install pennylane-sf`                   |
| Google Cirq                       | `pip install pennylane-cirq`                 |
| Rigetti forest                    | `pip install pennylane-forest`               |
| Microsoft Q#                      | `pip install pennylane-qsharp`               |

**Install all of the above at once**
```bash
pip install \
  pennylane-sf \
  pennylane-qiskit \
  amazon-braket-pennylane-plugin \
  pennylane-cirq \
  pennylane-forest \
  pennylane-qsharp
```

<hr>

## Installing interfaces

These interfaces seamlessly integrate various ML libraries with Pennylane
More can be found here: https://pennylane.ai/plugins.html

| Interface  name | Install                            |
| :-------------- | :--------------------------------- |
| NumPy/Autograd  | `pip install autograd`             |
| TensorFlow      | `pip install "tensorflow>=1.13.2"` |
| JAX             | `pip install jax jaxlib`           |
**Install all of the above at once**
```bash
pip install \
  autograd \
  "tensorflow>=1.13.2" \
  jax \
  jaxlib
```

| Interface  name | Install                                    |
| :-------------- | :----------------------------------------- |
| PyTorch         | `pip install torch torchvision torchaudio` |

<hr>

## Installing other tools

| Name       | Install                  |
| :--------- | :----------------------- |
| Matplotlib | `pip install matplotlib` |
| Pandas     | `pip install pandas`     |

<hr>