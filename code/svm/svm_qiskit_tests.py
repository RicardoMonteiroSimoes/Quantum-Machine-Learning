from typing import Sequence
import pythonlib.helpers as Helpers
import os
from colorama import Fore, Back, Style
import numpy as numpy
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.utils import QuantumInstance
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from math import pi

# Quantum Circuits
N_WIRES = 4
N_LAYERS = 2

# dummy features
features = [1.2345] * N_WIRES

# wires and devices
wires = range(N_WIRES)

# QC1


# def qml_circuit_qiskit_01(n_wires=2, n_layers=1):
#     feature_map = QuantumCircuit(n_wires)
#     ansatz = QuantumCircuit(n_wires)
#     for i in range(n_wires):
#         feature_map.ry(Parameter('i_{}'.format(str(i))), i)
#     feature_map.barrier()
#     for j in range(n_layers):
#         for k in range(n_wires):
#             ansatz.ry(Parameter('{}_w_{}'.format(str(j), str(k))), k)
#             ansatz.cry(Parameter('{}_w2_{}'.format(str(j), str(k))), k, (k+1) % n_wires)
#         if j != n_layers-1:
#             ansatz.barrier()

#     qc = QuantumCircuit(n_wires)
#     qc.append(feature_map, range(n_wires))
#     qc.append(ansatz, range(n_wires))
#     return qc.decompose().copy()


# circuit = qml_circuit_qiskit_01(N_WIRES, N_LAYERS)
# print("\nQC1")
# print(circuit.draw(vertical_compression='high', fold=-1, scale=0.5))

# circuit.draw('mpl', filename=os.getcwd() + '/code/svm/assets/circuit_01_qiskit.png')


# QC2
# def qml_circuit_qiskit_02(n_wires=2, n_layers=1):
#     feature_map = QuantumCircuit(n_wires)
#     ansatz = QuantumCircuit(n_wires)

#     for i in range(n_wires):
#         feature_map.ry(Parameter('i_{}'.format(str(i))), i)
#     feature_map.barrier()
#     for j in range(n_layers):
#         for k in range(n_wires):
#             ansatz.ry(Parameter('{}_w_{}'.format(str(j), str(k))), k)
#         for l in range(n_wires):
#             ansatz.cry(Parameter('{}_w2_{}'.format(str(j), str(l))), l, (l+1) % n_wires)
#         if j != n_layers-1:
#             ansatz.barrier()

#     qc = QuantumCircuit(n_wires)
#     qc.append(feature_map, range(n_wires))
#     qc.append(ansatz, range(n_wires))
#     return qc.decompose().copy()


# circuit = qml_circuit_qiskit_02(N_WIRES, N_LAYERS)
# print("\nQC2")
# print(circuit.draw(vertical_compression='high', fold=-1, scale=0.5))

# circuit.draw('mpl', filename=os.getcwd() + '/code/svm/assets/circuit_02_qiskit.png')

# QC 3


# def qml_circuit_qiskit_03(n_wires=2, n_layers=1):
#     feature_map = QuantumCircuit(n_wires)
#     ansatz = QuantumCircuit(n_wires)

#     for i in range(n_wires):
#         feature_map.ry(Parameter('i_{}'.format(str(i))), i)
#     feature_map.barrier()

#     for j in range(n_layers):
#         for k in range(n_wires):
#             ansatz.ry(Parameter('{}_w_{}'.format(str(j), str(k))), k)
#             ansatz.cz(k, (k+1) % n_wires)
#         if j != n_layers-1:
#             ansatz.barrier()

#     qc = QuantumCircuit(n_wires)
#     qc.append(feature_map, range(n_wires))
#     qc.append(ansatz, range(n_wires))
#     return qc.decompose().copy()


# circuit = qml_circuit_qiskit_03(N_WIRES, N_LAYERS)
# print("\nQC3")
# print(circuit.draw(vertical_compression='high', fold=-1, scale=0.5))

# circuit.draw('mpl', filename=os.getcwd() + '/code/svm/assets/circuit_03_qiskit.png')

# QC 4
# No entanglement
def qml_circuit_qiskit_04(n_wires=2, n_layers=1):
    """
    QC 4
    No entanglement
    """
    feature_map = QuantumCircuit(n_wires)
    ansatz = QuantumCircuit(n_wires)

    for i in range(n_wires):
        feature_map.ry(Parameter('i_{}'.format(str(i))), i)
    feature_map.barrier()

    for j in range(n_layers):
        for k in range(n_wires):
            ansatz.rx(Parameter('{}_w_x{}'.format(str(j), str(k))), k)
            ansatz.ry(Parameter('{}_w_y{}'.format(str(j), str(k))), k)
            ansatz.rz(Parameter('{}_w_z{}'.format(str(j), str(k))), k)
        if j != n_layers-1:
            ansatz.barrier()

    qc = QuantumCircuit(n_wires)
    qc.append(feature_map, range(n_wires))
    qc.append(ansatz, range(n_wires))
    return qc.decompose().copy()


circuit = qml_circuit_qiskit_04(N_WIRES, N_LAYERS)
print("\nQC4")
print(circuit.draw(vertical_compression='high', fold=-1, scale=0.5))

circuit.draw('mpl', filename=os.getcwd() + '/code/svm/assets/circuit_04_qiskit.png')


def qml_circuit_qiskit_05(n_wires=2, n_layers=1):
    """
    QC 5
    More rotation
    """
    feature_map = QuantumCircuit(n_wires)
    ansatz = QuantumCircuit(n_wires)

    for i in range(n_wires):
        feature_map.ry(Parameter('i_{}'.format(str(i))), i)
    feature_map.barrier()

    for j in range(n_layers):
        for k in range(n_wires):
            ansatz.rx(Parameter('{}_w_x{}'.format(str(j), str(k))), k)
            ansatz.ry(Parameter('{}_w_y{}'.format(str(j), str(k))), k)
            ansatz.rz(Parameter('{}_w_z{}'.format(str(j), str(k))), k)
            ansatz.cx(k, (k+1) % n_wires)
        if j != n_layers-1:
            ansatz.barrier()

    qc = QuantumCircuit(n_wires)
    qc.append(feature_map, range(n_wires))
    qc.append(ansatz, range(n_wires))
    return qc.decompose().copy()


circuit = qml_circuit_qiskit_05(N_WIRES, N_LAYERS)
print("\nQC5")
print(circuit.draw(vertical_compression='high', fold=-1, scale=0.5))

circuit.draw('mpl', filename=os.getcwd() + '/code/svm/assets/circuit_05_qiskit.png')
