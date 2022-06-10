#!/usr/bin/env python3

import pythonlib.qcircuits as qc
import os
import pickle
import numpy as np
# Qiskit Imports
import qiskit
from qiskit import IBMQ, Aer, QuantumCircuit, transpile, execute, assemble
from qiskit.circuit import Parameter
from qiskit.tools.monitor import job_monitor
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.ibmq import IBMQBackend, IBMQFactory
from qiskit.providers.ibmq.managed import IBMQJobManager
# sklearn
from sklearn.metrics import accuracy_score

# Settings
# change dir into script dir
abspath = os.path.abspath(__file__)
SCRIPT_DIRECTORY = os.path.dirname(abspath)
os.chdir(SCRIPT_DIRECTORY)

# VARS
DATASET_FILE = SCRIPT_DIRECTORY + '/../datasets/datasets2.data'  # 10 per dataset
NUMBER_DATASETS = 5
NUMBER_RUNS = 10
NUMBER_SAMPLES = 1000

load_dataset_args = (DATASET_FILE,
                     NUMBER_DATASETS,
                     NUMBER_RUNS,
                     NUMBER_SAMPLES)

q_job_tags = ['BA']


# IBM provider
IBM_PROVIDER = 'ibm-q'

# load IBMid account settings (e.g. access token) from `$HOME/.qiskit/qiskitrc`
IBMQ.load_account()

provider: IBMQFactory = IBMQ.get_provider(IBM_PROVIDER)
# Hardware choose from: 'ibmq_manila' 'ibmq_quito' 'ibmq_belem' 'ibmq_lima'
# backend_system = 'ibmq_manila'
# backend_system = 'ibmq_quito'
# backend_system = 'ibmq_belem'
# backend_system = 'ibmq_lima'

# Simulators: aer_simulator (local), ibmq_qasm_simulator (managed online)
# backend_system = 'aer_simulator'
backend_system = 'ibmq_qasm_simulator'

# Number of layers
N_LAYERS = os.getenv('QC_N_LAYERS', 2)

# Quantum circuits
quantum_circuits = {
    'qml_circuit_qiskit_01': qc.qml_circuit_qiskit_01,
    'qml_circuit_qiskit_02': qc.qml_circuit_qiskit_02,
    'qml_circuit_qiskit_03': qc.qml_circuit_qiskit_03,
    'qml_circuit_qiskit_04': qc.qml_circuit_qiskit_04,
    'qml_circuit_qiskit_05': qc.qml_circuit_qiskit_05,
}

# Optimizer
OPTIMIZER = {'BFGS': {'settings': 'maxfun=1000, maxiter=1500, ftol=2.220446049250313e-15, factr=None, iprint=- 1, epsilon=1e-08, eps=1e-08, options=None, max_evals_grouped=1'}}

# WEIGHTS
# add weights for best run (training score)
# Pretrained weights (best run)
pre_trained_weights = {
    'adhoc': {
        # score: 0.515
        'qml_circuit_qiskit_01': [35, np.array([-0.09036957, 0.36520811, 0.23494167, -0.42635724, 0.0906524, -0.11780418, 0.02135239, -0.23786084, 0.33658614, -0.10000176, -0.05779378, 0.03945693])],
        # score: 0.515
        'qml_circuit_qiskit_02': [36, np.array([-0.01138465, 0.36002186, 0.03402444, 0.05463159, -0.12911004, -0.31213203, -0.05238947, -0.3805265, 0.20104617, -0.3476284, 0.20315625, 0.33901702])],
        # score: 0.55
        'qml_circuit_qiskit_03': [36, np.array([-0.37336153, 0.29211136, -0.12508506, -0.20210956, 0.25441093, -0.14432376])],
        # score: 0.515
        'qml_circuit_qiskit_04': [36, np.array([-0.16230161, 0.08223786, -0.01971387, 0.71706785, 0.7744954, 0.18807107, 1.59273271, 1.24177171, 0.19475585, 0.80125989, 0.62050877, 0.04836066, 0.17955921, -0.32863574, -0.14579556, 0.40913576, 0.52741562, 0.63075515])],
        # score: 0.545
        'qml_circuit_qiskit_05': [35, np.array([0.76723692, 1.82757115, 0.46119638, 1.51145879, 1.4401381, 0.00839699, 0.69829498, 0.4367755, -0.09164764, 1.33776881, -0.01726824, 0.46476175, 1.61389319, 0.00246734, 0.01511495, 0.21603041, 0.45958991, 0.25382214])],
    },
    'custom': {
        # score: 0.525
        'qml_circuit_qiskit_01': [13, np.array([-0.58282155, 0.47875083, -0.96392689, -0.0701078, 0.01699089, 0.92916345, -0.44421655, 0.76865501])],
        # score: 0.535
        'qml_circuit_qiskit_02': [12, np.array([-0.38504194, 0.28468596, -0.23195013, -0.07552733, -0.27532117, 0.97145239, -1.09443134, 0.80431347])],
        # score: 0.525
        'qml_circuit_qiskit_03': [17, np.array([-0.30182622, 0.1731863, 0.093595, -0.07102579])],
        # score: 0.525
        'qml_circuit_qiskit_04': [17, np.array([0.09141497, -0.21979954, 0.23608025, 0.24070365, 0.58588813, 0.36342374, 0.14485474, 0.28186781, -0.39017232, -0.14056828, 0.62876466, 0.96281972])],
        # score: 0.58
        'qml_circuit_qiskit_05': [17, np.array([1.55106066, 1.23574958, 0.59315241, 0.68635426, 0.8589475, 1.08056338, -1.81005369, 1.37672194, 0.16152071, -0.25158786, 0.55612662, 0.72301435])],
    },
    'iris': {
        # score: 0.9333333333333333
        'qml_circuit_qiskit_01': [25, np.array([0.09720091, -0.14258517, -0.32437601, -0.57580837, 0.58556917, -0.541049, 0.52079736, 0.35170595, 0.04522364, -1.08775061, -0.6134527, -0.60029693, 0.27153155, 0.34823277, 1.25292275, 1.23930591])],
        # score: 0.9333333333333333
        'qml_circuit_qiskit_02': [27, np.array([-0.00420353, -0.374399, -0.48873756, -0.79846746, 0.57439902, 0.03146352, 1.23638586, 0.58019372, 0.19449422, -0.92451115, -0.31165707, -0.54901889, 0.26036608, -0.19344285, 0.55824307, 0.9824787])],
        # score: 0.8
        'qml_circuit_qiskit_03': [25, np.array([0.4967994, -0.2286168, 1.32390126, 0.53986672, 0.30355247, 0.09654148, 0.76218567, 0.41519104])],
        # score: 0.9666666666666667
        'qml_circuit_qiskit_04': [20, np.array([0.22570268, 0.39373317, 0.21044244, 0.295288, 0.10457663, 0.59157478, 1.05120622, 1.04121547, 0.39218911, 0.94822082, -0.19545515, 0.74122332, -0.28537614, 0.20837295, -0.15347311, -0.58770457, 0.09583234, -0.65200234, 0.56310227, 0.12168861, 0.89017455, 0.73883315, 0.53874533, 0.43689517])],
        # score: 0.9
        'qml_circuit_qiskit_05': [20, np.array([0.05733975, 1.04739408, 2.203076, 0.19329493, 0.72034665, -0.86530192, 0.83945666, 1.40359846, 0.56105784, 0.87337092, 1.11514034, 0.55075499, -0.09475539, 0.86596356, 1.01729052, 0.1358492, 0.04747434, 0.46076933, -0.07259315, 0.03283566, 0.36936315, 0.72526758, 0.72689192, 0.88413079])],
    },
    'rain': {
        # score: 0.695
        'qml_circuit_qiskit_01': [5, np.array([0.28678465, 0.14433704, 0.04298832, -0.01601313, -0.20120602, 0.11385418, -0.30997485, -0.14876871, -0.16197786, 0.15910639, -0.05245376, 0.02912868, 0.05544095, 0.02377168, -0.00301807, 0.07445739, 0.14995917, -0.04618472, 0.09852834, -0.16583264])],
        # score: 0.71
        'qml_circuit_qiskit_02': [6, np.array([-0.05806164, 0.19608212, -0.11383801, 0.22433648, -0.3147627, -0.23063359, -0.18122564, 0.08236058, -0.12350757, 0.19797048, 0.27052197, -0.03272731, 0.24710416, -0.17693411, 0.20885786, 0.42895732, 0.047435, -0.30331482, 0.07478585, -0.19657808])],
        # score: 0.73
        'qml_circuit_qiskit_03': [6, np.array([0.12024154, -0.10066185, -0.13007191, -0.05100274, -0.01748789, 0.23055275, -0.09000602, -0.05409799, -0.02766909, -0.03798971])],
        # score: 0.685
        'qml_circuit_qiskit_04': [6, np.array([0.09930154, -0.10275156, 0.29287532, -0.13552311, 0.15629054, 0.80582433, 0.41650366, -0.09926849, 0.4865807, -0.00762555, 1.15896181, 0.48881554, 0.42658482, 0.56782003, 0.36985975, 0.51323176, 0.31927304, -0.21552822, 0.36845993, -0.13562558, -0.35437811, -0.36762096, -0.14901681, -0.39806071, -0.03493228, 0.83312802, 0.1227823, 0.37031285, 0.56825931, 0.40631651])],
        # score: 0.4
        'qml_circuit_qiskit_05': [2, np.array([-3.36496396e-02, -2.48794185e-04, -8.58381591e-03, -4.69451419e-03, 5.82019354e-03, 7.20770205e-01, -4.65950822e-02, -5.68380420e-03, 1.09195885e-01, 7.99974334e-03, -2.62405488e-02, 4.82733493e-01, 2.28942496e-01, 7.00922227e-01, 4.34819506e-01, -2.46411708e-03, -2.99957453e-03, -4.72964138e-03, 3.27481793e-03, -6.21065831e-03, -2.77928466e-02, 6.71320619e-03, -4.64280799e-02, -2.23585198e-02, -1.32022659e-02, 5.91027346e-01, 7.73532648e-01, 9.63651728e-01, 3.71838271e-01, 4.40200807e-01])],
    },
    'vlds': {
        # score: 0.855
        'qml_circuit_qiskit_01': [44, np.array([0.21029785, -0.21085909, 0.00132721, -0.05558718, 0.36369162, 0.12271294, 0.1053352, 0.28205396, 0.06281035, -0.0097319, -0.22222805, -0.77329597, 0.05045812, 0.06997182, -0.45308729, -0.03980669, -0.11497727, 0.91146973, -0.09194671, 0.02769738])],
        # score: 0.825
        'qml_circuit_qiskit_02': [46, np.array([0.17334773, -0.7046343, 0.01537427, 0.34709756, 0.00177104, 0.04655366, -0.00791732, 0.60538502, -0.02567824, 0.2104666, -0.14785016, -0.39968697, 0.02817373, -0.3565756, 0.00811825, 0.02337454, -0.01048775, 0.57739096, 0.00263528, -0.18660432])],
        # score: 0.92
        'qml_circuit_qiskit_03': [46, np.array([0.09972664, -0.03467638, 0.64305684, -0.02133031, 0.0204905, -0.04057203, -0.00423946, 0.63254637, -0.01889261, -0.01012555])],
        # score: 0.77
        'qml_circuit_qiskit_04': [46, np.array([2.34725514e-01, 2.69590878e-02, -3.04791856e-01, -1.15003904e-01, 1.31455658e-01, 8.54548041e-02, -5.82360802e-02, 1.07270745e+00, 3.41127704e-01, 4.77440854e-01, 7.94206296e-02, 5.10240347e-01, 1.08070003e+00, 1.26497424e+00, 9.57134073e-01, -2.32663250e-01, -3.04920642e-02, 1.39910305e-01, 3.60347058e-01, 2.98263436e-01, -3.46379193e-02, 1.01901422e-03, 2.70146678e-01, 1.51363106e-03, -3.73434304e-01, 6.48205520e-01, 9.24438493e-01, 4.11468366e-01, 5.06797846e-01, 9.60099843e-01])],
        # score: 0.535
        'qml_circuit_qiskit_05': [47, np.array([1.61947646e+00, 2.18500140e+00, -2.94560852e-02, -1.69129803e-02, -4.62399691e-01, 7.25125763e-01, 1.57115844e+00, 3.10533903e+00, 1.28399797e-02, 1.99775193e+00, 1.63136954e+00, 6.59874462e-01, 4.74898915e-01, 5.46464635e-01, 1.05052559e+00, 8.84810646e-02, 1.42082971e-02, -3.42234161e-03, 5.88008046e-04, 1.18182164e+00, 1.56251147e+00, 3.25699311e-03, 6.37206622e-03, 8.85023833e-04, -2.87549090e-02, 9.58581935e-01, 7.33931649e-01, 1.56690576e-01, 6.06005450e-01, 1.90195741e-01])],
    },
}

# FUNCTIONS


def parity(x, o_shape=2):
    # print("parity x: {}".format(x))
    return '{:b}'.format(x).count('1') % o_shape


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
    # check if number of samples[train + test] is equal to number of samples as expected
    print("Verify datasets ...")
    for d in data_sets:
        if d[1] == 'iris':
            assert d[2][1].shape[0]+d[2][2].shape[0] == 150, f"data corruption detected for dataset: {d[1]}"
        else:
            assert d[2][1].shape[0]+d[2][2].shape[0] == NUMBER_SAMPLES, f"data corruption detected for dataset: {d[1]}"
    # check if number of dataset being generated is as expected
    assert len(data_sets) == NUMBER_DATASETS*NUMBER_RUNS

    print("done\n")
    return data_sets


# MAIN
if __name__ == '__main__':
    print("Running script in folder \"{}\"".format(SCRIPT_DIRECTORY))
    datasets = load_verify_datasets(load_dataset_args)

    print(f"Backend system: {backend_system}")

    scores = {}

    for dataset_name in pre_trained_weights.keys():
        for (circuit_name, values) in pre_trained_weights[dataset_name].items():
            (dataset_id, weights) = values
            score_key = f"{dataset_name}[{dataset_id}]-{circuit_name}"
            print(score_key)

            # get sample & label data
            dataset = datasets[dataset_id]
            (current_dataset_id, dataset_name, data) = dataset
            assert current_dataset_id == dataset_id, "dataset ids do not match."
            (_, sample_test, label_train, label_test) = data
            if dataset_name == 'iris':
                assert len(sample_test) == len(label_test) == 30, "test set: features and labels are not of equal length or not equal to 20."
            else:
                assert len(sample_test) == len(label_test) == 200, "test set: features and labels are not of equal length or not equal to 20."

            n_wires = len(sample_test[0])
            # count different values of targets (parity)
            OUTPUT_SHAPE = len(np.unique(label_train))

            # initialize classes array
            calculated_classes = np.zeros(len(sample_test))
            if backend_system == 'aer_simulator':
                backend = Aer.get_backend(backend_system)
            else:
                backend: IBMQBackend = provider.get_backend(backend_system)

            run_scores = np.zeros(NUMBER_RUNS)
            for run_index in range(NUMBER_RUNS):
                # Build cicruits
                circuits_array = []

                # Loop over all features an add builded circuits to array
                for index, input_feature_arr in enumerate(sample_test):
                    #print(f"input_features: [{input_feature_arr}]")
                    params = np.concatenate((weights, input_feature_arr), axis=0)
                    #print(f"params: [{params}]")
                    # build q circuit
                    quantum_circuit: QuantumCircuit = quantum_circuits[circuit_name](n_wires=n_wires, n_layers=N_LAYERS)
                    # bind weights and input features
                    circuit_with_params = quantum_circuit.bind_parameters(params)
                    #print(circuit_with_params.draw(vertical_compression='high', fold=-1, scale=0.5))
                    circuits_array.append(circuit_with_params)

                transpiled_circuits_array = transpile(circuits_array, backend=backend)

                job_tags = [
                    list(OPTIMIZER.keys())[0],
                    *q_job_tags,
                    *[dataset_name, circuit_name]
                ]
                job_name = f"{dataset_name}-{circuit_name}"

                if backend_system == 'aer_simulator':
                    results = qiskit.execute(circuits_array, backend=backend, shots=1024).result()
                else:
                  if dataset_name == 'iris':
                    job_manager = IBMQJobManager()
                    job_set = job_manager.run(transpiled_circuits_array, backend=backend, name=job_name, job_tags=job_tags)
                    results = job_set.results()
                  else:
                    job_manager = IBMQJobManager()
                    job_set = job_manager.run(transpiled_circuits_array[:100], backend=backend, name=job_name, job_tags=job_tags)
                    results = job_set.results()
                    job_manager2 = IBMQJobManager()
                    job_set2 = job_manager2.run(transpiled_circuits_array[100:], backend=backend, name=job_name+'2', job_tags=job_tags)
                    results2 = job_set2.results()

                if dataset_name == 'iris':
                  for transpiled_circuit_index in range(len(transpiled_circuits_array)):
                      counts = results.get_counts(transpiled_circuit_index)
                      # get max
                      max_score_register_value = max(counts, key=counts.get)
                      # convert to int
                      int_value_score = int(max_score_register_value, 2)
                      # determine class
                      calculated_classes[transpiled_circuit_index] = parity(int_value_score, OUTPUT_SHAPE)
                      #print(f"{counts} - {max_score_register_value} (integer_value: {int_value_score}) - {calculated_classes[transpiled_circuit_index]}")

                else:
                  for transpiled_circuit_index in range(len(transpiled_circuits_array[:100])):
                      counts = results.get_counts(transpiled_circuit_index)
                      # get max
                      max_score_register_value = max(counts, key=counts.get)
                      # convert to int
                      int_value_score = int(max_score_register_value, 2)
                      # determine class
                      calculated_classes[transpiled_circuit_index] = parity(int_value_score, OUTPUT_SHAPE)
                      #print(f"{counts} - {max_score_register_value} (integer_value: {int_value_score}) - {calculated_classes[transpiled_circuit_index]}")

                      counts = results2.get_counts(transpiled_circuit_index)
                      # get max
                      max_score_register_value = max(counts, key=counts.get)
                      # convert to int
                      int_value_score = int(max_score_register_value, 2)
                      # determine class
                      calculated_classes[transpiled_circuit_index] = parity(int_value_score, OUTPUT_SHAPE)
                      #print(f"{counts} - {max_score_register_value} (integer_value: {int_value_score}) - {calculated_classes[transpiled_circuit_index]}")

                # calculate average
                # print(pd.DataFrame({'true label_test': label_test, 'predicted labels': calculated_classes})) # print arrays side by side
                current_circuit_score = accuracy_score(label_test, calculated_classes)
                run_scores[run_index] = current_circuit_score

                # print result for current dataset and circuit
                print(f"{dataset_name} {circuit_name} ACCURACY_SCORE: {current_circuit_score}")

            # add scores to dict
            scores[score_key] = np.average(run_scores)
            print("")

    print(f"\nOptimizer: {OPTIMIZER}")
    print(f"\nscores: {scores}")
