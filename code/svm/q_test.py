import os

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from pennylane.templates import StronglyEntanglingLayers, AngleEmbedding

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATASET = os.getcwd() + '/code/datasets/iris_classes1and2_scaled.txt'

num_qubits = 4
num_layers = 2


# dev = qml.device("default.qubit", wires=2)

# def get_angles(x):
#     beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
#     beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
#     beta2 = 2 * np.arcsin(
#         np.sqrt(x[2] ** 2 + x[3] ** 2)
#         / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
#     )

#     return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

# def statepreparation(a):
#     qml.RY(a[0], wires=0)

#     qml.CNOT(wires=[0, 1])
#     qml.RY(a[1], wires=1)
#     qml.CNOT(wires=[0, 1])
#     qml.RY(a[2], wires=1)

#     qml.PauliX(wires=0)
#     qml.CNOT(wires=[0, 1])
#     qml.RY(a[3], wires=1)
#     qml.CNOT(wires=[0, 1])
#     qml.RY(a[4], wires=1)
#     qml.PauliX(wires=0)


# def layer(W):
#     qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
#     qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
#     qml.CNOT(wires=[0, 1])


# @qml.qnode(dev)
# def circuit(weights, angles):
#     statepreparation(angles)

#     for W in weights:
#         layer(W)

#     return qml.expval(qml.PauliZ(0))


# MY Quantum Circuit
wires = range(num_qubits)
q_dev = qml.device('default.qubit', wires=wires)
q_dev_02 = qml.device('default.qubit', wires=wires)
# get the shape of the StronglyEntanglingLayers
shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)


@qml.qnode(q_dev, diff_method="parameter-shift")
def circuit(feature_vector, parameters):
    """A variational quantum model."""
    for i in list(wires):
        qml.Hadamard(wires=i)
    # embedding
    AngleEmbedding(features=feature_vector, wires=wires, rotation='Y')
    # trainable measurement
    StronglyEntanglingLayers(weights=parameters, wires=wires, ranges=[1]*num_layers, imprimitive=qml.ops.CNOT)
    return qml.expval(qml.PauliZ(0))
# -------------------------


def variational_classifier(weights, bias, feature_vector):
    return circuit(feature_vector, weights) + bias


def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def cost(weights, bias, features, labels):
    predictions = [variational_classifier(weights, bias, f) for f in features]
    return square_loss(labels, predictions)


# ------- iris
dataset = load_iris()

# targets: 3 classes => [0,...,1,...,2,...]
targets = dataset['target']
Y = targets-1  # set Y data to -1,0,1
Y = np.where(Y[:100] == 0, Y[:100]+1, Y[:100])
# print("Y", Y)
# print("data len Y", len(Y))


data = dataset['data']
# print("dataset data", data)
X = data[:100]
X_orig = X
X = StandardScaler().fit_transform(X)
X_scaled = np.array(MinMaxScaler(feature_range=(-1, 1), copy=True).fit_transform(X), requires_grad=False)

# print("data len X_scaled", len(X_scaled))
# print("data type X_scaled", type(X_scaled))

# -------
# data = np.loadtxt(DATASET)
# X = data[:, 0:2]

# # pad the vectors to size 2^2 with constant values
# padding = 0.3 * np.ones((len(X), 1))
# X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
# print("First X sample (padded)    :", X_pad[0])

# # normalize each input
# normalization = np.sqrt(np.sum(X_pad ** 2, -1))
# X_norm = (X_pad.T / normalization).T
# print("First X sample (normalized):", X_norm[0])

# # angles for state preparation are new features
# features = np.array([get_angles(x) for x in X_norm], requires_grad=False)
# print("First features sample      :", features[0])

# Y = data[:, -1]

# -------
np.random.seed(0)
num_data = len(Y)
num_train = int(0.8 * num_data)
index = np.random.permutation(range(num_data))
feats_train = X_scaled[index[:num_train]]
Y_train = Y[index[:num_train]]
feats_val = X_scaled[index[num_train:]]
Y_val = Y[index[num_train:]]

# We need these later for plotting
X_train = X_scaled[index[:num_train]]
X_val = X_scaled[index[num_train:]]

weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

opt = NesterovMomentumOptimizer(0.01)
batch_size = 5

# train the variational classifier
weights = weights_init
bias = bias_init
for it in range(20):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,))
    feats_train_batch = feats_train[batch_index]
    Y_train_batch = Y_train[batch_index]
    # print("RUN with: ", weights, bias, feats_train_batch, Y_train_batch)
    (weights, bias, _, _), _cost = opt.step_and_cost(cost, weights, bias, feats_train_batch, Y_train_batch)

    # Compute predictions on train and validation set
    predictions_train = [np.sign(variational_classifier(weights, bias, f)) for f in feats_train]
    predictions_val = [np.sign(variational_classifier(weights, bias, f)) for f in feats_val]

    # Compute accuracy on train and validation set
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    print(
        "Iter: {:5d} | Cost: {:0.7f} ({:0.7f}) | Acc train: {:0.7f} | Acc validation: {:0.7f} "
        "".format(it + 1, cost(weights, bias, X_scaled, Y), _cost, acc_train, acc_val)
    )


# plt.figure()
# cm = plt.cm.RdBu

# h = 0.03
# offset = 0.1
# x_min_scaled, x_max_scaled = X_scaled[:, 0].min() - offset, X_scaled[:, 0].max() + offset
# y_min_scaled, y_max_scaled = X_scaled[:, 1].min() - offset, X_scaled[:, 1].max() + offset

# # make data for decision regions
# xx, yy = np.meshgrid(np.arange(x_min_scaled, x_max_scaled, h), np.arange(y_min_scaled, y_max_scaled, h))
# # X_grid = [np.array([x, y]) for x, y in zip(xx.flatten(), yy.flatten())]

# #print("X_grid", X_grid)

# Z = np.array(
#     [variational_classifier(weights, bias, f) for f in np.c_[xx.ravel(), yy.ravel()]]
# )
# Z = Z.reshape(xx.shape)

# print("Z", Z)
# divnorm = colors.TwoSlopeNorm(vmin=Z.min(), vcenter=0, vmax=Z.max())
# # print("levels", levels)

# # plot decision regions
# cnt = plt.contourf(
#     xx, yy, Z, norm=divnorm, levels=[-1, 0, 1], cmap=cm, alpha=0.8, extend="both"
# )
# # plt.contour(
# #     xx, yy, Z, levels=[0.0], colors=("black",), linestyles=("--",), linewidths=(0.8,)
# # )
# plt.colorbar(cnt, ticks=[-1, 0, 1])

# # plot data
# plt.scatter(
#     X_train[:, 0][Y_train == 1],
#     X_train[:, 1][Y_train == 1],
#     c="b",
#     marker="o",
#     edgecolors="k",
#     label="class 1 train",
# )
# plt.scatter(
#     X_val[:, 0][Y_val == 1],
#     X_val[:, 1][Y_val == 1],
#     c="b",
#     marker="^",
#     edgecolors="k",
#     label="class 1 validation",
# )
# plt.scatter(
#     X_train[:, 0][Y_train == -1],
#     X_train[:, 1][Y_train == -1],
#     c="r",
#     marker="o",
#     edgecolors="k",
#     label="class -1 train",
# )
# plt.scatter(
#     X_val[:, 0][Y_val == -1],
#     X_val[:, 1][Y_val == -1],
#     c="r",
#     marker="^",
#     edgecolors="k",
#     label="class -1 validation",
# )

# plt.legend()
# plt.show()
