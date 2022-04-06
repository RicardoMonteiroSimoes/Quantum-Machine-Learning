from pathlib import Path
import os
# pennylane
import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers, AngleEmbedding
from pennylane.optimize import NesterovMomentumOptimizer
from pennylane import broadcast
from pennylane import numpy as np
# qiskit
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.utils import QuantumInstance
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
# import numpy as np
import csv
import pickle
from math import pi
from datetime import datetime
# parallel
# import concurrent.futures

#  Helper Funtions


def load_data(filename):
    '''
    Dataset loading
    '''
    with open(filename, 'rb') as filehandle:
        # read the data as binary data stream
        return pickle.load(filehandle)


def verify_datasets_integrity(data_sets, number_datasets, number_samples, number_runs):
    # check if number of samples[train + test] is equal to number of samples as expected
    for d in data_sets:
        assert d[2][1].shape[0]+d[2][2].shape[0] == number_samples, "data corruption detected"
    # check if number of dataset being generated is as expected
    assert len(data_sets) == number_datasets*number_runs


'''
ML functions
'''


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss


# def cost(vc_classifier, weights, bias, features, labels):
#     predictions = [vc_classifier(weights, bias, f) for f in features]
#     return square_loss(labels, predictions)


def accuracy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)
    return loss


def get_classifier(circuit, n_features=2):
    output_shape = 2  # binary classification

    def parity(x):
        #print("parity x: {}".format(x))
        return '{:b}'.format(x).count('1') % output_shape

    q_simulator = Aer.get_backend('aer_simulator')

    # print("feature count: {}".format(n_features))

    # q_simulator.set_options(device='GPU')
    quantum_instance = QuantumInstance(q_simulator, shots=1024)

    circuit_qnn = CircuitQNN(circuit=circuit,
                             input_params=circuit.parameters[-n_features:],
                             weight_params=circuit.parameters[:-n_features],
                             interpret=parity,
                             output_shape=output_shape,
                             quantum_instance=quantum_instance)

    # construct classifier
    return NeuralNetworkClassifier(neural_network=circuit_qnn,
                                   optimizer=COBYLA())

# Write into file


def write_entry(target_file, header, entry, mode=['w', 'a'][1]):
    '''
    e.g, header=['device','dataset','dataset-id','feature map','score']
    Note: specify file to wirete under section '02 Global Parameters'
    '''
    # add header only if file does not exist
    target_file = Path(target_file)
    if not target_file.is_file():
        with open(target_file, mode) as data_file:
            data_file_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_file_writer.writerow(header)

    with open(target_file, mode) as data_file:
        data_file_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_file_writer.writerow(entry)


# run
def run_qiskit(dataset, circuit, n_layers=1):
    (sample_train, sample_test, label_train, label_test) = dataset
    N_WIRES = len(sample_train[0])  # count features e.g. wires (feature on each wire)

    classifier = get_classifier(circuit.copy(), N_WIRES)
    score_train, score_test = fit_and_score(classifier, sample_train, sample_test, label_train, label_test)

    return score_train, score_test


def fix_label_values(label_data):
    if(np.any(label_data[:] == 0)):
        np.subtract(label_data, 1, out=label_data, where=label_data == 0)
    if(np.any(label_data[:] == 2)):
        np.subtract(label_data, 1, out=label_data, where=label_data == 2)
    return label_data


def qkesvm(dataset, backend, n_layers=1):
    '''
    dataset sample_train,sample_test,label_train,label_test
    '''

    batch_size = 5
    iterations = 50

    (sample_train, sample_test, label_train, label_test) = dataset
    X = np.concatenate([sample_test, sample_train], requires_grad=False)

    # print("\nsample_test", sample_test.shape)
    # print("sample_train", sample_train.shape)

    # scale labels
    # print("label_train", label_train)
    label_train = fix_label_values(label_train)
    label_test = fix_label_values(label_test)
    Y = np.concatenate([label_test, label_train])
    # print("Y:", Y)

    N_WIRES = len(sample_train[0])  # count features

    ## weights and bias
    init_bias = np.array(0.0, requires_grad=True)

    # get the shape of the StronglyEntanglingLayers
    # shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=N_WIRES) # QC (testing)
    # init_weights = 0.01 * np.random.random(size=shape, requires_grad=True) # QC (testing)

    # init_weights = np.random.randn(n_layers, N_WIRES, 2, requires_grad=True)  # QC 1
    # init_weights = np.random.randn(n_layers, 2, N_WIRES, requires_grad=True)  # QC 2
    init_weights = np.random.randn(n_layers, N_WIRES, 1, requires_grad=True)  # QC 3

    # get circuit
    # circuit, vc_classifier, q_device = qml_circuit_01(N_WIRES)  # QC 1
    # circuit, vc_classifier, q_device = qml_circuit_02(N_WIRES)  # QC 2
    circuit, vc_classifier, q_device = qml_circuit_03(N_WIRES)  # QC 3

    print(qml.draw(circuit, expansion_strategy='device')([9.9]*N_WIRES, init_weights))

    def cost(weights, bias, features, labels):
        predictions = [vc_classifier(weights, bias, f) for f in features]
        return square_loss(labels, predictions)

    costs = np.zeros(iterations)
    weights = init_weights
    bias = init_bias
    opt = NesterovMomentumOptimizer(0.01)

    for it in range(iterations):

        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, len(sample_train), (batch_size,))
        feats_train_batch = np.array(sample_train[batch_index], requires_grad=False)
        label_train_batch = np.array(label_train[batch_index], requires_grad=False)
        # print("RUN: ", weights, bias, feats_train_batch, label_train_batch)
        (weights, bias, _, _), opt_costs = opt.step_and_cost(cost, weights, bias, feats_train_batch, label_train_batch)

        # Compute predictions on train and validation set
        predictions_train = [np.sign(vc_classifier(weights, bias, f)) for f in sample_train]
        predictions_val = [np.sign(vc_classifier(weights, bias, f)) for f in sample_test]
        predictions_all = [np.sign(vc_classifier(weights, bias, f)) for f in X]

        # Compute accuracy on train and validation set
        acc_train = accuracy(label_train, predictions_train)
        acc_val = accuracy(label_test, predictions_val)
        acc_all = accuracy(Y, predictions_all)

        # gather informations for plotting
        costs[it] = cost(weights, bias, X, Y)

        print(
            "Iter: {:5d} | Cost: {:0.7f} ({:0.7f})  | Acc train: {:0.7f} | Acc val: {:0.7f} | Acc all: {:0.7f} "
            "".format(it + 1, costs[it], opt_costs, acc_train, acc_val, acc_all)
        )

    run = []

    # kernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)

    # # step 1 -> calculate kernel matrices

    # kernel_matrisample_train = kernel.evaluate(x_vec=sample_train)
    # kernel_matrix_test = kernel.evaluate(x_vec=sample_test, y_vec=sample_train)

    # csvc = SVC(kernel='precomputed')

    # # step 2 -> train
    # csvc.fit(kernel_matrisample_train, label_train)

    # # step 3 -> test
    # score = csvc.score(kernel_matrix_test, label_test)

    # # COLLECTING RESULTS
    # run.append(score)
    # if(include_kernel):
    #     run.append(kernel_matrisample_train)
    #     run.append(kernel_matrix_test)
    #     run.append(kernel)

    # todo: fix this
    return run


'''
schema: <set-id,dataset-name,datapoints>
'''


def get_set_id(dataset):
    return dataset[0]


def run_experiment(data_sets, backend, targetfile='', number_runs=1):
    # fm_selector = {'z': 'ZFeatureMap', 'zz': 'ZZFeatureMap', 'p': 'PauliFeatureMap'}
    # ent_selector = {'l': 'linear', 'c': 'circular', 'f': 'full'}
    no_csv = True
    # HEADER = ['device', 'dataset', 'dataset-id', 'number features', 'feature map name', 'feature map depth', 'feature map entanglement', 'score']

    if targetfile == '':
        print('NO .CSV mode activated. Provide file name.')
    else:
        print('.CSV mode activated.')
        no_csv = False
    # print(str(HEADER))
    exp_data = list()
    for epoch in range(number_runs):
        for dataset in data_sets:
            set_id = get_set_id(dataset)
            # for feature_map in feature_maps[set_id]:
            #     if feature_maps_selector is None or feature_map.name in [fm_selector[sel] for sel in feature_maps_selector]:
            #         if entanglement is None or feature_map.entanglement in [ent_selector[sel] for sel in entanglement]:

            # device name
            print(backend[0], end=' ')
            entry = [backend[0], ]  # add device name

            # dataset name
            print(dataset[1], end=' ')
            entry.append(dataset[1])  # add dataset name

            # dataset id
            print(dataset[0], end=' ')
            entry.append(dataset[0])  # add dataset id

            # # number features
            # print(feature_map.num_qubits, end=' ')
            # entry.append(feature_map.num_qubits)  # number features

            # # feature map name
            # print(feature_map.name, end=' ')
            # entry.append(feature_map.name)  # add feature map name

            # # depth
            # print(feature_map.reps, end=' ')
            # entry.append(feature_map.reps)  # add feature map depth

            # # entanglement
            # if feature_map.name == 'ZFeatureMap':
            #     print('none', end=' ')
            #     entry.append('none')  # add feature map entanglement
            # else:
            #     print(feature_map.entanglement, end=' ')
            #     entry.append(feature_map.entanglement)  # add feature map entanglement
            print('\n')

            N_LAYERS = 5

            # run = qkesvm(dataset[2], backend[1], N_LAYERS)
            run = run_qiskit(dataset[2], backend[1], N_LAYERS)
            entry = np.concatenate((entry, run))  # add data from experiment
            exp_data.append(entry)
            print(entry[-1])
            # append entry to the file
            # if not no_csv:
            #     write_entry(targetfile, HEADER, entry, mode='a')
        #     else:
        #         print('ignored: ', feature_map.name, 'due to ', feature_map.entanglement, ' entanglement.')
        # else:
        #     print('ignored: ', feature_map.name, ', because was not in the white list.')
    return exp_data


def create_markdown_file(prefix='run', markdown='# Title\n') -> str:

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    filepath = os.getcwd() + '/runs/{}_{}.md'.format(prefix, timestamp)

    f = open(filepath, "w")
    f.write(markdown)
    f.close()

    return filepath


def runCircuits(data_sets, n_layers=1):

    N_LAYERS = n_layers
    circuits = [
        qml_circuit_qiskit_01,
        qml_circuit_qiskit_02,
        qml_circuit_qiskit_03,
    ]

    markdown = "# QNN data run (SVM Quantum Kernel)\n\n"
    current_dataset_name = ''

    # loop over datasets
    for dataset in data_sets:
        (dataset_id, dataset_name, data) = dataset
        print('Running dataset "{}" (id: {}) with {} layers'.format(dataset_name, dataset_id, N_LAYERS))

        if current_dataset_name != dataset_name:
            # dataset name
            markdown += '## {} (id: {})\n\n'.format(dataset_name, dataset_id)

        # (sample_train, sample_test, label_train, label_test) = data
        N_WIRES = len(data[0][0])  # count features e.g. wires (feature on each wire)

        def doCircuit(qml_circuit):
            circuit = qml_circuit(n_wires=N_WIRES, n_layers=N_LAYERS)

            circuit_name = qml_circuit.__name__
            print("{}".format(circuit_name), end=' ')
            # create plot
            # circuit_plot_filename = os.getcwd() + '/runs/assets/{}-{}-{}_layers.png'.format(dataset_name, circuit_name, N_LAYERS)
            # circuit.draw('mpl', filename=circuit_plot_filename, scale=0.5)

            # run circuit
            score_train, score_test = run_qiskit(data, circuit, N_LAYERS)
            # add to markdown
            # circuit_plot_basename = os.path.basename(circuit_plot_filename)
            # markdown += '[<img src="assets/{}" alt="{}" height="200px" />](assets/{})\n\n'.format(
            #     circuit_plot_basename, circuit_plot_basename, circuit_plot_basename)
            # markdown += 'Score train: {}\nScore test: {}\n\n'.format(score_train, score_test)
            return "{}:\nscore train: {} | score test: {}\n\n".format(circuit_name, score_train, score_test)

        print("→ parallel run:", end=' ')

        # doCircuitRun.main()

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = []

        #     for qml_circuit in circuits:
        #         futures.append(executor.submit(doCircuit, qml_circuit))
        #     for future in concurrent.futures.as_completed(futures):
        #         print(future.result(), end='\n')

        markdown += '\n'
        # set current dataset name
        current_dataset_name = dataset_name
        print("\n")

    markdown += '\n\nEOF\n'
    #filepath = create_markdown_file('run_{}-layers'.format(N_LAYERS), markdown)

    return 'done'  # 'Run has been saved to file: {}'.format(filepath.replace(os.getcwd()+'/', ''))
