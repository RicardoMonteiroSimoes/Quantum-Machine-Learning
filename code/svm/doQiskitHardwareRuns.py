#!/usr/bin/env python3

import pythonlib.qcircuits as qc
import os
import pickle
import numpy as np
# Qiskit Imports
from qiskit import IBMQ, Aer, QuantumCircuit, transpile, execute, assemble
from qiskit.circuit import Parameter
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy
# sklearn
from sklearn.metrics import accuracy_score

# Settings
# change dir into script dir
abspath = os.path.abspath(__file__)
SCRIPT_DIRECTORY = os.path.dirname(abspath)
os.chdir(SCRIPT_DIRECTORY)

# VARS
DATASET_FILE = SCRIPT_DIRECTORY + '/../datasets/datasets_10.data'  # 10 per dataset
# DATASET_FILE = SCRIPT_DIRECTORY + '/../datasets/datasets.data' # 13 per dataset
NUMBER_DATASETS = 5
NUMBER_RUNS = 13
NUMBER_SAMPLES = 100

load_dataset_args = (DATASET_FILE,
                     NUMBER_DATASETS,
                     NUMBER_RUNS,
                     NUMBER_SAMPLES)

# OUTPUT_SHAPE (number of classes)
OUTPUT_SHAPE = 2

# Quantum Settings
N_QUBITS = 5
# IBM provider
IBM_PROVIDER = 'ibm-q'

# load IBMid account settings (e.g. access token) from `$HOME/.qiskit/qiskitrc`
IBMQ.load_account()

# determine least busy backend
provider = IBMQ.get_provider(IBM_PROVIDER)
# backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits == N_QUBITS and
#                                        not x.configuration().simulator and x.status().operational == True))

# Number of layers
N_LAYERS = os.getenv('QC_N_LAYERS', 2)

# Quantum circuits
quantum_circuits = [
    ('qml_circuit_qiskit_01', qc.qml_circuit_qiskit_01),
    ('qml_circuit_qiskit_02', qc.qml_circuit_qiskit_02),
    ('qml_circuit_qiskit_03', qc.qml_circuit_qiskit_03),
    ('qml_circuit_qiskit_04', qc.qml_circuit_qiskit_04),
    ('qml_circuit_qiskit_05', qc.qml_circuit_qiskit_05)
]

# WEIGHTS #
# Pretrained weights (best run)
pre_trained_weights = {
    'adhoc': {  # dataset name and id
        'qml_circuit_qiskit_01': np.array([-0.10511535, 0.64954487, 1.93420391, -0.50899113, 0.46932607, -0.21376011,
                                           0.36373938, 1.5038279, 1.73870895, 2.24519027, -0.6743587, 0.84247449])
    }
}

# FUNCTIONS


def parity(x):
    # print("parity x: {}".format(x))
    return '{:b}'.format(x).count('1') % OUTPUT_SHAPE


def load_data(filename):
    """
    Load pickle file from given filename including path
    """
    with open(filename, 'rb') as filehandle:
        # read the data as binary data stream
        return pickle.load(filehandle)


def load_verify_datasets(args):
    """
    Load datasets as pickle file from disk
    """
    (DATASET_FILE, NUMBER_DATASETS, NUMBER_RUNS, NUMBER_SAMPLES) = args
    print("Loading datasets ...")
    data_sets = load_data(DATASET_FILE)
    # hlp.verify_datasets_integrity(data_sets, number_datasets=NUMBER_DATASETS,
    #                               number_samples=NUMBER_SAMPLES, number_runs=NUMBER_RUNS)
    print("done\n")
    return data_sets


# for testing
# backend = Aer.get_backend('aer_simulator')
backend = provider.get_backend('ibmq_qasm_simulator')

# print("backends:", backend)

# MAIN
if __name__ == '__main__':
    print("Running script in folder \"{}\"".format(SCRIPT_DIRECTORY))
    datasets = load_verify_datasets(load_dataset_args)

    print("Running circuits width \"{}\" ...".format(backend))
    # Use filtered datasets (13) like: `for index, dataset in enumerate([datasets[i] for i in [1, 14, 27, 40, 53]]):`
    # Use filtered datasets (10) like: `for index, dataset in enumerate([datasets[i] for i in [1,11,21,31,41]]):`
    # for index, dataset in enumerate(datasets):
    for index, dataset in enumerate([datasets[i] for i in [1, 11, 21, 31, 41]]):
        (dataset_id, dataset_name, data) = dataset
        print("{}: {}".format(index, dataset[1]))
        (sample_train, sample_test, label_train, label_test) = data

        X = np.concatenate((sample_train, sample_test), axis=0)
        Y = np.concatenate((label_train, label_test), axis=0)
        np.subtract(Y, 1, out=Y, where=Y == 2)  # fix labels

        assert len(X) == len(Y), "features and labels not of equal length."

        try:
            if (pre_trained_weights[dataset_name]):

                for q_circuit in quantum_circuits:
                    (circuit_name, q_circ_builder) = q_circuit

                    try:
                        pre_trained_weights_current_dataset = pre_trained_weights[dataset_name][circuit_name]
                        print("found circ: {}".format(circuit_name))

                        print("weights:", pre_trained_weights_current_dataset)
                        n_wires = len(X[0])
                        print("n_wires:", n_wires)

                        # initialize classes array
                        calculated_classes = np.zeros(len(X))

                        # build q circuit
                        quantum_circuit: QuantumCircuit = q_circ_builder(n_wires=n_wires, n_layers=N_LAYERS)

                        # Loop over all features
                        for index, input_feature_arr in enumerate(X):
                            print("input_features:", input_feature_arr)
                            params = np.concatenate((pre_trained_weights_current_dataset, input_feature_arr), axis=0)
                            circuit_with_params = quantum_circuit.bind_parameters(params)
                            # print(circuit_with_params.draw(vertical_compression='high', fold=-1, scale=0.5))
                            job = execute(circuit_with_params, backend)
                            result = job.result()
                            counts = result.get_counts()
                            # get max
                            max_score_register_value = max(counts, key=counts.get)
                            # convert to int
                            int_value_score = int(max_score_register_value, 2)
                            # get class
                            calculated_classes[index] = parity(int_value_score)

                        print("classes:", calculated_classes)
                        # average
                        print("accuracy_score", accuracy_score(Y, calculated_classes))

                    except:
                        print("Skipping circuit: {}".format(circuit_name))
                        pass

        except:
            print("Skipping dataset: {}".format(dataset_name))
            pass

    # dummy_input_features = np.array([1.0, 2.0, 3.0])

    # quantum_circuit = quantum_circuits[0](n_wires=3, n_layers=N_LAYERS)

    # params = np.concatenate((pre_trained_weights['adhoc_29']['qml_circuit_qiskit_01'],
    #                          dummy_input_features), axis=0)
    # print(pre_trained_weights['adhoc_29']['qml_circuit_qiskit_01'])
    # print(params)

    # circuit_with_data = quantum_circuit.bind_parameters(params)
    # print(circuit_with_data.draw(vertical_compression='high', fold=-1, scale=0.5))
    # job = execute(circuit_with_data, backend)

    # alternative to execute:
    # mapped_circuit = transpile(circuit_with_data, backend=backend)
    # qobj = assemble(mapped_circuit, backend=backend, shots=1024)
    # job = backend.run(qobj)

    # result = job.result()
    # counts = result.get_counts()
    # print("counts", counts)
    # try:
    #   import matplotlib.pyplot as plt
    #   plot_histogram(counts, title='counts')
    #   plt.show()
    # except Exception as e:
    #   print("Error", e)
