import multiprocessing
import pprint
# import pythonlib.helpers as hlp
import pythonlib.qcircuits as qc
import os
import pickle
import numpy as np
# qiskit
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# VARS
DATASET_FILE = os.getcwd() + '/code/datasets/datasets.data'
NUMBER_DATASETS = 5
NUMBER_RUNS = 13
NUMBER_SAMPLES = 100

load_dataset_args = (DATASET_FILE,
                     NUMBER_DATASETS,
                     NUMBER_RUNS,
                     NUMBER_SAMPLES)
# Q Cicruit settings
N_LAYERS = 2

# Q circuits
quantum_circuits = [
    qc.qml_circuit_qiskit_01,
    qc.qml_circuit_qiskit_02,
    qc.qml_circuit_qiskit_03
]

optimizers = [
    (COBYLA(), 'cobyla'),
]


def get_classifier(circuit: QuantumCircuit, _weights: list, n_features=2):
    output_shape = 2  # binary classification

    def parity(x):
        # print("parity x: {}".format(x))
        return '{:b}'.format(x).count('1') % output_shape

    def callback(weights, obj_func_eval):
        # _weights[0] = weights
        _weights.append(weights)

    q_simulator = Aer.get_backend('aer_simulator')

    try:
        import GPUtil
        if(len(GPUtil.getGPUs()) > 0):
            q_simulator.set_options(device='GPU')
            print("GPU device option for qiskit simulator has been set")
    except:
        print("Failed to set qiskit simulator device option: GPU")
    quantum_instance = QuantumInstance(q_simulator, shots=1024)

    circuit_qnn = CircuitQNN(circuit=circuit,
                             input_params=circuit.parameters[-n_features:],
                             weight_params=circuit.parameters[:-n_features],
                             interpret=parity,
                             output_shape=output_shape,
                             quantum_instance=quantum_instance)

    # construct classifier
    return NeuralNetworkClassifier(neural_network=circuit_qnn,
                                   callback=callback,
                                   optimizer=COBYLA())


def worker_datasets(return_dict, dataset):
    """thread worker function"""
    (dataset_id, dataset_name, data) = dataset
    N_WIRES = len(data[0][0])  # is also feature count

    qcirc_results = []

    for q_circ in quantum_circuits:
        circuit_name = q_circ.__name__
        weights = []
        classifier = get_classifier(q_circ(n_wires=N_WIRES, n_layers=N_LAYERS).copy(), weights, N_WIRES)

        (sample_train, sample_test, label_train, label_test) = data
        np.subtract(label_train, 1, out=label_train, where=label_train == 2)
        np.subtract(label_test, 1, out=label_test, where=label_test == 2)

        # fit classifier to data
        classifier.fit(sample_train, label_train)
        score_train = classifier.score(sample_train, label_train)
        score_test = classifier.score(sample_test, label_test)
        qcirc_results.append((circuit_name,  score_train, score_test, weights[-1]))

    # print("res", score_train, score_test)
    return_dict[dataset_name] = (dataset_id, qcirc_results)


def load_data(filename):
    with open(filename, 'rb') as filehandle:
        # read the data as binary data stream
        return pickle.load(filehandle)


def load_verify_datasets(args):
    (DATASET_FILE, NUMBER_DATASETS, NUMBER_RUNS, NUMBER_SAMPLES) = args
    print("Loading datasets ...")
    data_sets = load_data(DATASET_FILE)
    # hlp.verify_datasets_integrity(data_sets, number_datasets=NUMBER_DATASETS,
    #                               number_samples=NUMBER_SAMPLES, number_runs=NUMBER_RUNS)
    print("done\n")
    return data_sets


if __name__ == '__main__':
    datasets = load_verify_datasets(load_dataset_args)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    print("Running circuits ...")
    # Todo: remove filtered datasets | [datasets[i] for i in [1, 14, 27, 40, 53]]
    for dataset in [datasets[i] for i in [1, 14]]:
        p = multiprocessing.Process(target=worker_datasets, args=(return_dict, dataset))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    print("results: ")
    pprint.pprint(return_dict.items())
