import multiprocessing
import pythonlib.qcircuits as qc
import os
import re
import pickle
import numpy as np
from datetime import datetime
# qiskit
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# change dir into script dir
abspath = os.path.abspath(__file__)
SCRIPT_DIRECTORY = os.path.dirname(abspath)
os.chdir(SCRIPT_DIRECTORY)

# VARS
DATASET_FILE = SCRIPT_DIRECTORY + '/../datasets/datasets.data'
NUMBER_DATASETS = 5
NUMBER_RUNS = 13
NUMBER_SAMPLES = 100

load_dataset_args = (DATASET_FILE,
                     NUMBER_DATASETS,
                     NUMBER_RUNS,
                     NUMBER_SAMPLES)
# Q Cicruit settings
# N_LAYERS => Defines the number of layers for the quantum circuits
N_LAYERS = os.getenv('QC_N_LAYERS', 2)

# Q circuits
quantum_circuits = [
    qc.qml_circuit_qiskit_01,
    qc.qml_circuit_qiskit_02,
    qc.qml_circuit_qiskit_03
]

# generated circuit for plots
generated_quantum_circuits = []

optimizers = [
    (COBYLA(), 'cobyla'),
]


def get_classifier(circuit: QuantumCircuit, _weights: list, n_features=2):
    output_shape = 2  # binary classification

    def parity(x):
        # print("parity x: {}".format(x))
        return '{:b}'.format(x).count('1') % output_shape

    def callback(weights, obj_func_eval):
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


def worker_datasets(return_list: dict, dataset):
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

        if(circuit_name not in qcirc_results):
            qcirc_results[circuit_name] = []

        # get the generated quantum circuit
        quantum_circuit = q_circ(n_wires=N_WIRES, n_layers=N_LAYERS).copy()

        # get the generated classifier
        classifier = get_classifier(quantum_circuit, weights, N_WIRES)

        (sample_train, sample_test, label_train, label_test) = data
        np.subtract(label_train, 1, out=label_train, where=label_train == 2)
        np.subtract(label_test, 1, out=label_test, where=label_test == 2)

        # fit classifier to data
        classifier.fit(sample_train, label_train)
        score_train = classifier.score(sample_train, label_train)
        score_test = classifier.score(sample_test, label_test)

        qcirc_results[circuit_name].append([score_train, score_test, np.array(weights[-1])])

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
    (DATASET_FILE, NUMBER_DATASETS, NUMBER_RUNS, NUMBER_SAMPLES) = args
    print("Loading datasets ...")
    data_sets = load_data(DATASET_FILE)
    # hlp.verify_datasets_integrity(data_sets, number_datasets=NUMBER_DATASETS,
    #                               number_samples=NUMBER_SAMPLES, number_runs=NUMBER_RUNS)
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


def arr_to_str(arr):
    """
    Array to string helper for score (train, test) and weights array
    """
    str_1 = '[{}]'.format(','.join([x.strip(' \n\r,][') for x in re.split("\s", str(arr[0])) if re.match(r"^\[?[-+]?[0-9]*\.?[0-9]+\]?,?$", x)]))
    str_2 = '[{}]'.format(','.join([x.strip(' \n\r,][') for x in re.split("\s", str(arr[1])) if re.match(r"^\[?[-+]?[0-9]*\.?[0-9]+\]?,?$", x)]))
    return '`{}`, `{}`'.format(str_1, str_2)


def plot_and_save_circuits(dataset_name, n_wires, timestamp=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")):
    """
    Plot and save the quantum circuits from the circuits array
    """
    circuit_plots = []
    # Loop over circuits
    for circuit_builder in quantum_circuits:
        circuit_name = circuit_builder.__name__
        circuit = circuit_builder(n_wires=n_wires, n_layers=N_LAYERS)
        circuit_plots.append(
            ('assets/{}-{}-({},{})-{}.png'.format(dataset_name, circuit_name,
                                                  n_wires, N_LAYERS,
                                                  timestamp),
             circuit_name, dataset_name))
        circuit.draw('mpl', filename=SCRIPT_DIRECTORY + "/runs/" + circuit_plots[-1][0])

    return circuit_plots


def generate_markdown_from_list(result_list):
    """
    Generate markdown file from result_list array. Save the markdown info file
    """
    if(len(result_list) <= 0):
        print("empty result")
        return

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    average_dict = {}
    per_run_dict = {}
    n_average = {
        'custom': 0,
        'iris': 0,
        'adhoc': 0,
        'rain': 0,
        'vlds': 0,
    }
    circ_plots = []

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
            # create circuit plots
            for plot_item in plot_and_save_circuits(dataset_name, n_wires, timestamp):
                circ_plots.append(plot_item)
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

    markdown = "# QNN data run\n\n"

    # Circuit plots
    markdown += "## Quantum Circuits\n"
    markdown += "Quantum Circuits plots for each dataset\n"
    markdown += "| dataset | circuit | plot |\n"
    markdown += "| :-----: | :-----: | :--: |\n"
    for (relative_filepath, circuit_name, dataset_name) in circ_plots:
        markdown += "| {} | {} | <img src=\"{}\" alt=\"{}\" /> |\n".format(dataset_name, circuit_name, relative_filepath, circuit_name)
    markdown += '\n\n'

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
            if (index == 0):
                md_header_cols = '| dataset name and run |'
                md_structure_cols = '| :----------: |'
                for col in range(len(run_value)):
                    md_header_cols += ' circuit-{:02d}: score (train, test) and weights  |'.format(col)
                    md_structure_cols += ' :--------: |'
                markdown += md_header_cols + "\n"
                markdown += md_structure_cols + "\n"
            if run_key.startswith(key):
                run_value1 = arr_to_str(run_value[0])
                run_value2 = arr_to_str(run_value[1])
                run_value3 = arr_to_str(run_value[2])
                markdown += "| `{}` | {} | {} | {} |\n".format(run_key, run_value1, run_value2, run_value3)

        markdown += '\n\n'

    filepath = save_markdown_to_file('training_run', markdown, timestamp)
    print('Run has been saved to file: {}'.format(filepath))


# MAIN
if __name__ == '__main__':
    print("Running script in folder \"{}\"".format(SCRIPT_DIRECTORY))
    datasets = load_verify_datasets(load_dataset_args)

    manager = multiprocessing.Manager()
    return_list = manager.list()
    jobs = []

    print("Running circuits ...")
    # Use filtered datasets like: `for dataset in [datasets[i] for i in [1, 14, 27, 40, 53]]:`
    for index, dataset in enumerate(datasets):
        print("start process {}".format(index))
        p = multiprocessing.Process(target=worker_datasets, args=(return_list, dataset))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    print("results: ", return_list)
    # sort by dataset name (fisrt) and dataset id (second)
    return_list.sort(key=sortDatasetsByNameAndId, reverse=False)

    generate_markdown_from_list(return_list)
