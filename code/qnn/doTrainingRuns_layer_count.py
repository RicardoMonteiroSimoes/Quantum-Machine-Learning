#!/usr/bin/env python3

import multiprocessing
import pythonlib.qcircuits as qc
import os
import re
import pickle
import numpy as np
from datetime import datetime
# qiskit
from qiskit.algorithms.optimizers import COBYLA, ADAM, SPSA, L_BFGS_B
from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# change dir into script dir
abspath = os.path.abspath(__file__)
SCRIPT_DIRECTORY = os.path.dirname(abspath)
os.chdir(SCRIPT_DIRECTORY)

# VARS
DATASET_FILE = SCRIPT_DIRECTORY + '/../datasets/datasets.data'
NUMBER_DATASETS = 5
NUMBER_RUNS = 10
NUMBER_SAMPLES = 100

load_dataset_args = (DATASET_FILE,
                     NUMBER_DATASETS,
                     NUMBER_RUNS,
                     NUMBER_SAMPLES)

# Q circuits
quantum_circuits = [
    qc.qml_circuit_qiskit_01,
    qc.qml_circuit_qiskit_02,
    qc.qml_circuit_qiskit_03,
    qc.qml_circuit_qiskit_04,
    qc.qml_circuit_qiskit_05,
]

"""
Optimizers:

SPSA - defaults:maxiter=100, blocking=False, allowed_increase=None, trust_region=False, learning_rate=None, perturbation=None, last_avg=1, resamplings=1, perturbation_dims=None, second_order=False, regularization=None, hessian_delay=0, lse_solver=None, initial_hessian=None, callback=None, termination_checker=None
=> https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.SPSA.html#qiskit.algorithms.optimizers.SPSA

COBYLA - defaults: maxiter=1000, disp=False, rhobeg=1.0, tol=None, options=None, **kwargs
=> https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.COBYLA.html#qiskit.algorithms.optimizers.COBYLA

ADAM - defaults: maxiter=10000, tol=1e-06, lr=0.001, beta_1=0.9, beta_2=0.99, noise_factor=1e-08, eps=1e-10, amsgrad=False, snapshot_dir=None
=> https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.ADAM.html#qiskit.algorithms.optimizers.ADAM

L_BFGS_B - defaults:maxfun=1000, maxiter=15000, ftol=2.220446049250313e-15, factr=None, iprint=- 1, epsilon=1e-08, eps=1e-08, options=None, max_evals_grouped=1
=> https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.L_BFGS_B.html#qiskit.algorithms.optimizers.L_BFGS_B

...etc. => https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.html

Read also: https://arxiv.org/pdf/2106.08682.pdf
"""
# change the optimizer here
optimizer = (COBYLA(maxiter=1000), 'COBYLA')
# optimizer = (ADAM(maxiter=1000,amsgrad=True), 'ADAM AMSGRAD')
# optimizer = (SPSA(maxiter=100), 'SPSA')
# optimizer = (L_BFGS_B(maxiter=1500), 'BFGS')


def get_classifier(circuit: QuantumCircuit, _weights: list, _obj_func_eval: list, n_features=2):
    output_shape = 2  # binary classification

    def parity(x):
        # print("parity x: {}".format(x))
        return '{:b}'.format(x).count('1') % output_shape

    def callback(weights, obj_func_eval):
        _obj_func_eval.append(obj_func_eval)
        _weights.append(weights)

    # Qiskit backend
    q_simulator = Aer.get_backend('aer_simulator')
    # use legacy simulator:
    # q_simulator = BasicAer.get_backend('qasm_simulator')

    # try GPU support if available (needs `qiskit-aer-gpu` package)
    try:
        import GPUtil
        if(len(GPUtil.getGPUs()) > 0):
            q_simulator.set_options(device='GPU')
            print("GPU enabled")
    except:
        print("Failed to set GPU support")

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
                                   optimizer=optimizer[0])


def worker_datasets(return_list: dict, dataset, n_layers=2):
    """
    Thread worker function (multipocessing):
    Gets the classifiers for all the quantum circuits and trains them with the given dataset and optimizer.
    Saves the results into return_list
    """
    (dataset_id, dataset_name, data) = dataset
    N_WIRES = len(data[0][0])  # is also feature count

    qcirc_results = {}

    # Loop over circuits
    for q_circ in quantum_circuits:
        circuit_name = q_circ.__name__
        weights = []
        obj_func_eval = []

        if(circuit_name not in qcirc_results):
            qcirc_results[circuit_name] = []

        # get the generated quantum circuit
        quantum_circuit: QuantumCircuit = q_circ(n_wires=N_WIRES, n_layers=n_layers).copy()

        # get the generated classifier
        classifier = get_classifier(quantum_circuit, weights, obj_func_eval, N_WIRES)

        (sample_train, sample_test, label_train, label_test) = data
        # ugly hack to fix iris labels. Don't do this at home
        np.subtract(label_train, 1, out=label_train, where=label_train == 2)
        np.subtract(label_test, 1, out=label_test, where=label_test == 2)

        # fit classifier to data
        classifier.fit(sample_train, label_train)
        score_train = classifier.score(sample_train, label_train)
        score_test = classifier.score(sample_test, label_test)

        qcirc_results[circuit_name].append([score_train, score_test, np.array(weights[-1]), obj_func_eval])

    return_list.append([(dataset_name, dataset_id, N_WIRES), qcirc_results])


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
    print("Loading and verifying dataset ...")

    (DATASET_FILE, NUMBER_DATASETS, NUMBER_RUNS, NUMBER_SAMPLES) = args
    data_sets = load_data(DATASET_FILE)
    # check if number of samples[train + test] is equal to number of samples as expected
    for d in data_sets:
        assert d[2][1].shape[0]+d[2][2].shape[0] == NUMBER_SAMPLES, "data corruption detected"
    # check if number of dataset being generated is as expected
    assert len(data_sets) == NUMBER_DATASETS*NUMBER_RUNS

    print("done\n")
    return data_sets


def save_markdown_to_file(filename_prefix='run', markdown='# Title\n', timestamp=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) -> str:
    """
    Save given markdown to file
    """
    filepath = SCRIPT_DIRECTORY + '/runs/{}_{}.md'.format(filename_prefix, timestamp)
    f = open(filepath, "w")
    f.write(markdown)
    f.close()
    return filepath


def sortDatasetsByNameAndId(item):
    """
    Return first element of item which contains a tuple:
    (dataset_name, dataset_id)
    """
    return item[0]


def arr_to_str(arr, show_weights=True):
    """
    Array to string helper for score (train, test) and weights array
    """
    str_1 = '[{}]'.format(','.join([x.strip(' \n\r,][') for x in re.split("\s", str(arr[0])) if re.match(r"^\[?[-+]?[0-9]*\.?[0-9e+-]+\]?,?$", x)]))
    if show_weights:
        str_2 = '[{}]'.format(','.join([x.strip(' \n\r,][') for x in re.split("\s", str(arr[1])) if re.match(r"^\[?[-+]?[0-9]*\.?[0-9e+-]+\]?,?$", x)]))
        return '`{}`, `{}`'.format(str_1, str_2)
    return '`{}`'.format(str_1)


def generate_markdown_from_list(result_list, n_layers):
    """
    Generate markdown file from result_list array. Save the markdown info file
    """
    if(len(result_list) <= 0):
        print("empty result")
        return

    average_dict = {}
    per_run_dict = {}
    n_average = {
        'custom': 0,
        'iris': 0,
        'adhoc': 0,
        'rain': 0,
        'vlds': 0,
    }

    print('\nPrepare data\n')
    # loop over dataset results
    for item in result_list:
        print("dataset info: ", item[0])
        dataset_name, dataset_id, n_wires = item[0]
        n_average[dataset_name] += 1  # Count occurences to calculate average
        dataset_name_id = '{}_{}'.format(dataset_name, dataset_id)

        # create entry for dataset if it does not exist
        if(dataset_name not in average_dict):
            average_dict[dataset_name] = []
        if(dataset_name_id not in per_run_dict):
            per_run_dict[dataset_name_id] = []

        # loop over circuit runs
        for i, (key, value) in enumerate(item[1].items()):
            arr = average_dict[dataset_name]
            run_arr = per_run_dict[dataset_name_id]

            score_train, score_test, weights = value[0]

            np_weights = np.array(weights, dtype=np.float64)
            run_arr.append([[score_train, score_test], np_weights])

            np_scores = np.array([score_train, score_test], dtype=np.float64)

            try:
                arr[i] = [np.add(arr[i][0], np_scores), np.add(arr[i][1], np_weights)]
            except IndexError:
                arr = arr.append([np_scores, np_weights])

    markdown = "# Quantum Neural Network Classifier run\n\n"
    markdown += "**Settings:**\n"
    markdown += "Used Optimizer for Neural Network Classifier: `{}`\n".format(optimizer[1])
    markdown += "Layer count: `{}`\n\n".format(n_layers)

    for (key, value) in average_dict.items():
        markdown += "## {}\n".format(key)
        markdown += '#### Average\n'

        markdown += "| circuit | ø score train | ø score test | ø weights |\n"
        markdown += "| ------: | :-----------: | :----------: | :-------: |\n"

        for index, circuit in enumerate(value):
            circ1_str = str(circuit[0][0]/n_average[key])
            circ2_str = str(circuit[0][1]/n_average[key])
            circ3_str = '[{}]'.format(','.join([
                x.strip(' \n\r][') for x in re.split("\s",
                                                     str(circuit[1]/n_average[key])) if re.match(r"^\[?[-+]?[0-9]*\.?[0-9]+\]?,?$", x)
            ]))
            markdown += "| circuit-{:02d} | `{}` | `{}` | `{}` |\n".format(index, circ1_str, circ2_str, circ3_str)
        markdown += '\n\n'

        markdown += '#### Per run data\n'
        for index, (run_key, run_value) in enumerate(per_run_dict.items()):
            # table header
            if (index == 0):
                md_header_cols = '| dataset name and run |'
                md_structure_cols = '| :----------: |'
                for col in range(len(run_value)):
                    md_header_cols += ' circuit-{:02d}: score (train, test)  |'.format(col)
                    md_structure_cols += ' :--------: |'
                markdown += md_header_cols + "\n"
                markdown += md_structure_cols + "\n"
            # table body
            if run_key.startswith(key):
                markdown_row = "| `{}` |".format(run_key)
                for run_data in run_value:
                    markdown_row += " {} |".format(arr_to_str(run_data, show_weights=False))
                markdown += markdown_row + "\n"

        markdown += '\n\n'

    return markdown


# MAIN
if __name__ == '__main__':
    print("Running script in folder \"{}\"".format(SCRIPT_DIRECTORY))
    datasets = load_verify_datasets(load_dataset_args)

    manager = multiprocessing.Manager()
    jobs = []
    markdown = ""

    # Do the following layers
    N_LAYERS_ARR = [1, 2, 4, 8]

    print("Running circuits ...")
    for n_layers in N_LAYERS_ARR:
        return_list = manager.list()

        pool = multiprocessing.Pool()
        pool.starmap(worker_datasets, [(return_list, datasets[i], n_layers) for i in range(40, 50, 1)])
        pool.close()
        pool.join()

        # sort by dataset name (first) and dataset id (second)
        return_list.sort(key=sortDatasetsByNameAndId, reverse=False)

        markdown += generate_markdown_from_list(return_list, n_layers)

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    filepath = save_markdown_to_file('layer_counts_training_run', markdown, timestamp)
    print('Run has been saved to file: {}'.format(filepath))
