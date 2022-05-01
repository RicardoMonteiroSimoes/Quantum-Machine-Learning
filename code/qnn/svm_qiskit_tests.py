#!/usr/bin/env python3

import os
import numpy as np
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.circuit import Parameter
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


def qml_circuit_qiskit_01(n_wires=2, n_layers=1):
    feature_map = QuantumCircuit(n_wires)
    ansatz = QuantumCircuit(n_wires)
    for i in range(n_wires):
        feature_map.ry(Parameter('i_{}'.format(str(i))), i)
    feature_map.barrier()
    for j in range(n_layers):
        for k in range(n_wires):
            ansatz.ry(Parameter('{}_w_{}'.format(str(j), str(k))), k)
            ansatz.cry(Parameter('{}_w2_{}'.format(str(j), str(k))), k, (k+1) % n_wires)
        if j != n_layers-1:
            ansatz.barrier()

    qc = QuantumCircuit(n_wires)
    qc.append(feature_map, range(n_wires))
    qc.append(ansatz, range(n_wires))
    qc.measure_all()

    # return qc.copy()
    return qc.decompose().copy()


circuit = qml_circuit_qiskit_01(3, N_LAYERS)
print("\nQC1")
print(circuit.draw(output='text', vertical_compression='high', fold=-1, scale=0.5))
print(circuit.draw(output='latex_source', vertical_compression='high', fold=-1, scale=0.5))

# circuit.draw('mpl', filename=os.getcwd() + '/code/svm/assets/circuit_01_qiskit.png')

# input features weights binding
# print("circuit.parameters", circuit.parameters)

# input_features = np.array([1.0, 2.0, 3.0])
# pretrained_weights = np.array([-0.10511535, 0.64954487, 1.93420391, -0.50899113, 0.46932607, -0.21376011,
#                                0.36373938, 1.5038279, 1.73870895, 2.24519027, -0.6743587, 0.84247449])
# params = np.concatenate((pretrained_weights, input_features), axis=0)

# print("params", params)
# circuit_with_data = circuit.bind_parameters(params)

# print(circuit_with_data.draw(vertical_compression='high', fold=-1, scale=0.5))
# END: input features weights binding

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
# def qml_circuit_qiskit_04(n_wires=2, n_layers=1):
#     """
#     QC 4
#     No entanglement
#     """
#     feature_map = QuantumCircuit(n_wires)
#     ansatz = QuantumCircuit(n_wires)

#     for i in range(n_wires):
#         feature_map.ry(Parameter('i_{}'.format(str(i))), i)
#     feature_map.barrier()

#     for j in range(n_layers):
#         for k in range(n_wires):
#             ansatz.rx(Parameter('{}_w_x{}'.format(str(j), str(k))), k)
#             ansatz.ry(Parameter('{}_w_y{}'.format(str(j), str(k))), k)
#             ansatz.rz(Parameter('{}_w_z{}'.format(str(j), str(k))), k)
#         if j != n_layers-1:
#             ansatz.barrier()

#     qc = QuantumCircuit(n_wires)
#     qc.append(feature_map, range(n_wires))
#     qc.append(ansatz, range(n_wires))
#     return qc.decompose().copy()


# circuit = qml_circuit_qiskit_04(N_WIRES, N_LAYERS)
# print("\nQC4")
# print(circuit.draw(vertical_compression='high', fold=-1, scale=0.5))

# circuit.draw('mpl', filename=os.getcwd() + '/code/svm/assets/circuit_04_qiskit.png')


# def qml_circuit_qiskit_05(n_wires=2, n_layers=1):
#     """
#     QC 5
#     More rotation
#     """
#     feature_map = QuantumCircuit(n_wires)
#     ansatz = QuantumCircuit(n_wires)

#     for i in range(n_wires):
#         feature_map.ry(Parameter('i_{}'.format(str(i))), i)
#     feature_map.barrier()

#     for j in range(n_layers):
#         for k in range(n_wires):
#             ansatz.rx(Parameter('{}_w_x{}'.format(str(j), str(k))), k)
#             ansatz.ry(Parameter('{}_w_y{}'.format(str(j), str(k))), k)
#             ansatz.rz(Parameter('{}_w_z{}'.format(str(j), str(k))), k)
#             ansatz.cx(k, (k+1) % n_wires)
#         if j != n_layers-1:
#             ansatz.barrier()

#     qc = QuantumCircuit(n_wires)
#     qc.append(feature_map, range(n_wires))
#     qc.append(ansatz, range(n_wires))
#     return qc.decompose().copy()


# circuit = qml_circuit_qiskit_05(N_WIRES, N_LAYERS)
# print("\nQC5")
# print(circuit.draw(vertical_compression='high', fold=-1, scale=0.5))

# circuit.draw('mpl', filename=os.getcwd() + '/code/svm/assets/circuit_05_qiskit.png')
