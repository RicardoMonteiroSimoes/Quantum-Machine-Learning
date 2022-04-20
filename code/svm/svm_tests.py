from typing import Sequence
import pythonlib.helpers as Helpers
import os
from colorama import Fore, Back, Style
import numpy as numpy
from pennylane import numpy as np
import pennylane as qml
from pennylane import broadcast
from pennylane.templates import StronglyEntanglingLayers, AngleEmbedding
from pennylane.optimize import NesterovMomentumOptimizer
from pennylane.math import requires_grad
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from math import pi

# VARIABLES and CONSTANTS
NUMBER_OF_ITERATIONS_PER_RUN = 50
NUMBER_DATASETS = 5
NUMBER_RUNS = 13
NUMBER_SAMPLES = 100
DATASET_FILE = os.getcwd() + '/code/datasets/datasets.data'

# Quantum Circuits
N_WIRES = 4
N_LAYERS = 2

# dummy features
features = [1.2345] * N_WIRES

# wires and devices
wires = range(N_WIRES)
dev = qml.device('default.qubit', wires=wires)
dev_02 = qml.device('default.qubit', wires=wires)
dev_03 = qml.device('default.qubit', wires=wires)
dev_04 = qml.device('default.qubit', wires=wires)

# drawer style
qml.drawer.use_style('black_white')

# -------------------------
# Q:1


@qml.qnode(dev, diff_method="parameter-shift")
def circuit(feature_vector, parameters, wires):
    """A variational quantum model."""
    for i in list(wires):
        qml.Hadamard(wires=i)
    # embedding
    AngleEmbedding(features=feature_vector, wires=wires, rotation='Y')
    # trainable measurement
    StronglyEntanglingLayers(weights=parameters, wires=wires, ranges=[1]*N_LAYERS, imprimitive=qml.ops.CNOT)
    return qml.expval(qml.PauliZ(0))

# init_weights = (0.01 * np.random.randn(1, len(wires), 3), 0.0)
# init_weights = 0.01 * np.random.random((len(wires), 3)


# get the shape of the StronglyEntanglingLayers
shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_WIRES)
init_weights = np.random.random(size=shape, requires_grad=True)

print("\n# QC1\n")
print(init_weights)

print(qml.draw(circuit, expansion_strategy='device')(features, init_weights, wires))

fig, ax = qml.draw_mpl(circuit, expansion_strategy='device')(features, init_weights, wires)
plt.savefig(os.getcwd() + '/code/svm/assets/circuit_01.png')

# -------------------------
# Q:2


@qml.qnode(dev_02, diff_method="parameter-shift")
def circuit_02(feature_vector, parameters, wires):
    """A variational quantum model."""

    if len(wires) <= 1:
        raise ValueError('At least two wires are required.')

    # template
    def q_template(param1, param2, wires):
        qml.RY(param1, wires=wires[0])
        qml.CRY(param2, wires=wires)
    # embedding
    AngleEmbedding(features=feature_vector, wires=wires, rotation='Y')
    # trainable measurement
    for weights in parameters:
        if len(wires) == 2:
            qml.RY(weights[0][0], wires=wires[0])
            qml.CRY(weights[0][1], wires=[0, 1])
            qml.RY(weights[1][0], wires=wires[1])
            qml.CRY(weights[1][1], wires=[1, 0])
        else:
            broadcast(unitary=q_template, pattern='ring', wires=wires, parameters=weights)

    return qml.expval(qml.PauliZ(0))


init_weights = np.random.randn(N_LAYERS, N_WIRES, 2, requires_grad=True)

print("\n# QC2\n")
print(init_weights)

print(qml.draw(circuit_02, expansion_strategy='device')(features, init_weights, wires))

fig, ax = qml.draw_mpl(circuit_02, expansion_strategy='device')(features, init_weights, wires)
plt.savefig(os.getcwd() + '/code/svm/assets/circuit_02.png')

# -------------------------
# Q:3


@qml.qnode(dev_03, diff_method="parameter-shift")
def circuit_03(feature_vector, parameters, wires):
    """A variational quantum model."""

    if len(wires) <= 1:
        raise ValueError('At least two wires are required.')

    # template
    def q_template(param, wires):
        qml.CRY(param, wires=wires)
    # embedding
    AngleEmbedding(features=feature_vector, wires=wires, rotation='Y')

    # trainable measurement
    for weights in parameters:
        if len(wires) == 2:
            broadcast(unitary=qml.RY, pattern="single", wires=wires, parameters=weights[0])
            qml.CRY(weights[1][0], wires=[0, 1])
            qml.CRY(weights[1][1], wires=[1, 0])
        else:
            broadcast(unitary=qml.RY, pattern="single", wires=wires, parameters=weights[0])
            broadcast(unitary=q_template, pattern='ring', wires=wires, parameters=weights[1])

    return qml.expval(qml.PauliZ(0))


# init_weights = np.random.randn(N_WIRES, 1, 1, requires_grad=True)
init_weights = np.random.randn(N_LAYERS, 2, N_WIRES, requires_grad=True)

print("\n# QC3\n")
print(init_weights)

print(qml.draw(circuit_03, expansion_strategy='device')(features, init_weights, wires))

fig, ax = qml.draw_mpl(circuit_03, expansion_strategy='device')(features, init_weights, wires)
plt.savefig(os.getcwd() + '/code/svm/assets/circuit_03.png')

# -------------------------
# Q:4


@qml.qnode(dev_04, diff_method="parameter-shift")
def circuit_04(feature_vector, parameters, wires):
    """A variational quantum model."""

    if len(wires) <= 1:
        raise ValueError('At least two wires are required.')

    # template
    def q_template(param, wires):
        qml.RY(param, wires=wires[0])
        qml.CZ(wires=wires)
    # embedding
    AngleEmbedding(features=feature_vector, wires=wires, rotation='Y')
    # trainable measurement
    for weights in parameters:
        if len(wires) == 2:
            qml.RY(weights[0][0], wires=wires[0])
            qml.CZ(wires=[0, 1])
            qml.RY(weights[1][0], wires=wires[1])
            qml.CZ(wires=[1, 0])
        else:
            broadcast(unitary=q_template, pattern='ring', wires=wires, parameters=weights)

    return qml.expval(qml.PauliZ(0))


print("\n# QC4\n")
# init_weights = np.random.random((N_WIRES, 1), requires_grad=True)
init_weights = np.random.randn(N_LAYERS, N_WIRES, 1, requires_grad=True)

print(init_weights)

print(qml.draw(circuit_04, expansion_strategy='device')(features, init_weights, wires))

fig, ax = qml.draw_mpl(circuit_04, expansion_strategy='device')(features, init_weights, wires)
plt.savefig(os.getcwd() + '/code/svm/assets/circuit_04.png')


# -------------------------


# ==================================================
# ==================================================
# LOAD PICKLE DATASET
# data_sets = Helpers.load_data(DATASET_FILE)
# # verify dataset
# Helpers.verify_datasets_integrity(data_sets, number_datasets=NUMBER_DATASETS,
#                                   number_samples=NUMBER_SAMPLES, number_runs=NUMBER_RUNS)

#
# for dataset in data_sets:

#     if dataset[1] == 'iris':
#         print("name: ", dataset[1])
#         print("datatset id: ", dataset[0])
#         (sample_train, sample_test, label_train, label_test) = dataset[2]
#         print("sample_train: ", sample_train[0],
#               "\nsample_test: ", sample_test[0],
#               "\nlabel_train: ", label_train[0],
#               "\nlabel_test: ", label_test[0])
