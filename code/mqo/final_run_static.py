from syslog import LOG_SYSLOG
from qiskit import *
from qiskit import Aer
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import numpy as np
import math
import random
import csv
import argparse
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pickle

import matplotlib.pyplot as plt
from matplotlib import cm


def main(argv):

    print('Static solver for MQO')
    print('---------------------------------------------------')
    print('Loading problem colection')
    #read pickle
    path = "runs/data/problems_with_solutions.p"
    data = pickle.load(open(path, "rb"))
    problems = scale_problems(data['problems'])
    print(problems)



    print('Creating circuit')
    circuit = create_circuit()
    print(circuit.draw())
    simulator_runs = []
    real_hardware_runs = []
    print('Starting simulator runs')
    for problem in problems:
        

    print('Finished execution.')
    print('---------------------------------------------------')
        
if __name__ == '__main__':
    import sys
    exit(main(sys.argv[1:]))