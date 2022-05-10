#!/usr/bin/env python3

import pythonlib.qcircuits as qc
import os
import pickle
import numpy as np
# Qiskit Imports
from qiskit import IBMQ, Aer, QuantumCircuit, transpile, execute, assemble
from qiskit.circuit import Parameter
from qiskit.tools.monitor import job_monitor
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.ibmq import IBMQBackend, IBMQFactory
# sklearn
from sklearn.metrics import accuracy_score

# Settings
# change dir into script dir
abspath = os.path.abspath(__file__)
SCRIPT_DIRECTORY = os.path.dirname(abspath)
os.chdir(SCRIPT_DIRECTORY)

# VARS
DATASET_FILE = SCRIPT_DIRECTORY + '/datasets.data'  # 10 per dataset
NUMBER_DATASETS = 5
NUMBER_RUNS = 10
NUMBER_SAMPLES = 100

load_dataset_args = (DATASET_FILE,
                     NUMBER_DATASETS,
                     NUMBER_RUNS,
                     NUMBER_SAMPLES)

q_job_tags = ['BA']

# OUTPUT_SHAPE (number of classes)
OUTPUT_SHAPE = 2

# Quantum Settings
N_QUBITS = 5
# IBM provider
IBM_PROVIDER = 'ibm-q'

# load IBMid account settings (e.g. access token) from `$HOME/.qiskit/qiskitrc`
IBMQ.load_account()

# get provider
provider: IBMQFactory = IBMQ.get_provider(IBM_PROVIDER)
backend_system = ''  # choose from: 'ibmq_manila' 'ibmq_quito' 'ibmq_belem' 'ibmq_lima'

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

# Optimizer
OPTIMIZER = {'ADAM AMSGRAD': {'settings': 'maxiter=1000, tol=1e-06, lr=0.001, beta_1=0.9, beta_2=0.99, noise_factor=1e-08, eps=1e-10, amsgrad=True, snapshot_dir=None'}}

# WEIGHTS
# add weights for best run (training score)
# Pretrained weights (best run)
pre_trained_weights = {
    'adhoc': {
        'qml_circuit_qiskit_01': np.array([]),  # score:
        'qml_circuit_qiskit_02': np.array([]),  # score:
        'qml_circuit_qiskit_03': np.array([]),  # score:
        'qml_circuit_qiskit_04': np.array([]),  # score:
        'qml_circuit_qiskit_05': np.array([]),  # score:
    },
    'custom': {
        'qml_circuit_qiskit_01': np.array([-0.29540564, 0.44252074, -0.6386595, -0.48140697, 0.68792073, 0.74256223, 0.30238779, -0.42532599]),  # score: 0.6
        'qml_circuit_qiskit_02': np.array([0.62104385, -0.32025427, -0.64281016, 0.21215402, 0.78684627, -0.12439845, -0.55650062, -0.15502515]),  # score: 0.55
        'qml_circuit_qiskit_03': np.array([-0.44742534, -0.22366666, -0.40683055, -0.21633652]),  # score: 0.55
        'qml_circuit_qiskit_04': np.array([0.08180276, -0.02626414, 0.14115286, -0.34184175, 0.21735755, 1.01737257, -0.06969894, 0.53363984, -0.05744692, -0.32770793, 0.18828458, 0.06141439]),  # score: 0.55
        'qml_circuit_qiskit_05': np.array([-0.24992258, 1.59572859, -0.66780293, 0.64863274, 0.01968169, 0.29459427, 1.09935585, 0.22656056, 1.29708391, 0.98843071, 0.59106936, 0.05584295]),  # score: 0.55
    },
    'iris': {
        # score: 1.0
        'qml_circuit_qiskit_01': np.array([-0.17157065, 1.21376712, 0.37442282, 0.85114773, 0.44744956, 0.09449407, 0.89491815, 1.33229949, 0.22001866, 1.19538293, 0.7357702, 1.28090361, 0.32739243, 0.11608572, 0.84917121, -0.19104296]),
        # score: 1.0
        'qml_circuit_qiskit_02': np.array([-0.28163945, 0.74731721, 0.809255, 1.15566414, 0.13731054, 0.50960219, 1.57455374, 0.65343114, 0.04491248, 0.97347177, 0.62106009, 1.08831996, 0.63596836, -0.14928698, 0.24740928, 0.10929216]),
        'qml_circuit_qiskit_03': np.array([0.19377634, 0.08409543, 0.65098321, 0.62654044, 0.16234158, 0.03066866, 0.60682621, 0.60329686]),  # score: 1.0
        'qml_circuit_qiskit_04': np.array([1.30649439, -0.23834209, 1.7076063, 0.18432451, 1.3910734, 0.70456464, 0.4608013, 1.20514393, -0.01405379, 0.54996844, -0.28173983, 0.4415028, -0.02648841, 0.39615597, 0.89538073, -0.47346727, 1.50160411, -0.23200959, 1.33789057, 1.10031745, 0.6254283, 0.58989277, 0.916514, 0.50278557]),  # score: 0.9
        'qml_circuit_qiskit_05': np.array([-0.00764106, 0.02590556, 1.28305328, 1.51250146, 0.25780792, 0.31238047, 0.15702906, 1.03305577, 0.93373024, 0.72343838, 1.58591745, 0.81074249, -0.01850049, 0.0340039, 1.26218088, 0.35602141, 0.06065977, 0.15066103, 0.69731337, 0.29356309, 0.7918182, 0.04991875, 0.22639765, 0.45439976]),  # score: 1.0
    },
    'rain': {
        'qml_circuit_qiskit_01': np.array([]),  # score:
        'qml_circuit_qiskit_02': np.array([]),  # score:
        'qml_circuit_qiskit_03': np.array([]),  # score:
        'qml_circuit_qiskit_04': np.array([]),  # score:
        'qml_circuit_qiskit_05': np.array([]),  # score:
    },
    'vlds': {
        'qml_circuit_qiskit_01': np.array([]),  # score:
        'qml_circuit_qiskit_02': np.array([]),  # score:
        'qml_circuit_qiskit_03': np.array([]),  # score:
        'qml_circuit_qiskit_04': np.array([]),  # score:
        'qml_circuit_qiskit_05': np.array([]),  # score:
    },
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


# MAIN
if __name__ == '__main__':
    print("Running script in folder \"{}\"".format(SCRIPT_DIRECTORY))
    datasets = load_verify_datasets(load_dataset_args)

    scores = {}

    # for index, dataset in enumerate([datasets[i] for i in [1, 11, 21, 31, 41]]):
    for index, dataset in enumerate(datasets):
        (dataset_id, dataset_name, data) = dataset
        print("{}: {}".format(index, dataset[1]))
        (sample_train, sample_test, label_train, label_test) = data

        X = np.concatenate((sample_train, sample_test), axis=0)
        Y = np.concatenate((label_train, label_test), axis=0)
        np.subtract(Y, 1, out=Y, where=Y == 2)  # fix labels

        assert len(X) == len(Y), "features and labels not of equal length."

        dataset_name_id_key = '{}_{}'.format(dataset_name, dataset_id)

        try:
            if (pre_trained_weights[dataset_name]):

                scores[dataset_name_id_key] = {}

                for q_circuit in quantum_circuits:
                    (circuit_name, q_circ_builder) = q_circuit

                    try:
                        pre_trained_weights_current_dataset = pre_trained_weights[dataset_name][circuit_name]
                        print("found circ: {}".format(circuit_name))

                        print("weights:", pre_trained_weights_current_dataset)
                        n_wires = len(X[0])
                        print("n_wires:", n_wires)

                        # initialize classes array
                        calculated_classes = np.zeros(len(sample_train))

                        # build q circuit
                        quantum_circuit: QuantumCircuit = q_circ_builder(n_wires=n_wires, n_layers=N_LAYERS)

                        # break at (testing)
                        break_at_index = 2

                        # Loop over all features
                        for index, input_feature_arr in enumerate(sample_train):
                            print("input_features:", input_feature_arr)
                            params = np.concatenate((pre_trained_weights_current_dataset, input_feature_arr), axis=0)
                            circuit_with_params = quantum_circuit.bind_parameters(params)
                            #print(circuit_with_params.draw(vertical_compression='high', fold=-1, scale=0.5))

                            # backend: IBMQBackend = provider.get_backend('ibmq_qasm_simulator')  # test
                            backend: IBMQBackend = provider.get_backend(backend_system)

                            job_tags = [
                                list(OPTIMIZER.keys())[0],
                                *q_job_tags,
                                *[dataset_name, circuit_name]
                            ]
                            job_name = f"{dataset_name} - {circuit_name}: {index}"

                            # job = execute(circuit_with_params, backend, shots=1024)
                            # alternative to execute (transpile and assemble the circuits yourself):
                            mapped_circuit = transpile(circuit_with_params, backend=backend)
                            job = backend.run(mapped_circuit, job_name=job_name, job_tags=job_tags, shots=1024)

                            # monitor job
                            job_monitor(job, interval=60)

                            result = job.result()
                            counts = result.get_counts()
                            # get max
                            max_score_register_value = max(counts, key=counts.get)
                            # convert to int
                            int_value_score = int(max_score_register_value, 2)
                            # determine class
                            calculated_classes[index] = parity(int_value_score)

                            # break out of for loop for testing
                            if index == break_at_index:  # 3 elements
                                break

                        # calculate average
                        current_circuit_overall_score = accuracy_score(label_train[:break_at_index+1], calculated_classes[:break_at_index+1])

                        # print result for current dataset and circuit
                        print(f"{dataset_name} {circuit_name} ACCURACY_SCORE : {current_circuit_overall_score}")

                        # add scores to dict
                        scores[dataset_name_id_key][circuit_name] = current_circuit_overall_score

                    except:
                        print("Skipping circuit: {}".format(circuit_name))
                        pass

        except:
            print("Skipping dataset: {}".format(dataset_name))
            pass

    print(f"\nOptimizer: {OPTIMIZER}")
    print(f"\nscores: {scores}")
