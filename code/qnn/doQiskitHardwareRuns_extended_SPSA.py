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
OPTIMIZER = {'SPSA': {'settings': 'maxiter=100, blocking=False, allowed_increase=None, trust_region=False, learning_rate=None, perturbation=None, last_avg=1, resamplings=1, perturbation_dims=None, second_order=False, regularization=None, hessian_delay=0, lse_solver=None, initial_hessian=None, callback=None, termination_checker=None'}}

# WEIGHTS
# add weights for best run (training score)
# Pretrained weights (best run)
pre_trained_weights = {
    'adhoc': {
        # score: 0.525
        'qml_circuit_qiskit_01': [34, np.array([1.21096083, 0.55771084, 0.71166418, 0.92417089, -0.3043202, -0.19376801, 0.27311397, 0.00345589, -0.13823056, 1.60673532, -0.34775185, 0.02434873])],
        # score: 0.545
        'qml_circuit_qiskit_02': [30, np.array([-0.48628242, 0.84534009, -0.92000718, -0.35210878, 0.18045309, -0.2997911, -0.32244289, -0.39383163, 1.38177147, -0.10491177, 0.49932572, 0.19853071])],
        # score: 0.545
        'qml_circuit_qiskit_03': [32, np.array([-7.09770016, 0.1379342, -6.3753666, 5.51633875, 6.42325733, -0.02620586])],
        # score: 0.515
        'qml_circuit_qiskit_04': [36, np.array([-0.46075185, -1.0372311, -0.36733042, 1.06824737, -1.11785489, 0.33578312, 2.54888089, -0.13366223, -0.67980072, 0.35302608, 0.71704628, 0.07408375, 1.05514879, 0.78905295, -0.45841718, 0.57954116, -1.38110821, -0.54466739])],
        # score: 0.575
        'qml_circuit_qiskit_05': [35, np.array([1.94459671e+00, 2.66901936e-01, -2.80997148e-03, 5.44720362e-01, 2.54535479e-01, -4.69377749e-02, -6.23143960e-02, -7.85206217e-01, -3.85723318e-01, 3.39518304e-01, -1.04606153e-02, -6.81967625e-03, 2.43619697e-01, 7.63968716e-03, 1.89857324e-03, 7.96978503e-01, 1.04243900e+00, 2.24289382e+00])],
    },
    'custom': {
        # score: 0.55
        'qml_circuit_qiskit_01': [19, np.array([0.99261994, 0.59345362, -0.12797849, -0.97962262, -0.9070245, 0.84656527, -0.9463901, 1.235442])],
        # score: 0.55
        'qml_circuit_qiskit_02': [17, np.array([1.07982388, -0.25763476, -0.59439824, 0.50062473, -1.56274497, -1.29399441, 1.21667324, -1.66401491])],
        # score: 0.525
        'qml_circuit_qiskit_03': [13, np.array([-2.34643837, 1.29716325, -1.93281745, -0.36963152])],
        # score: 0.525
        'qml_circuit_qiskit_04': [17, np.array([0.63448571, -1.45556331, 0.02514281, 0.04237618, 0.40381843, 0.97774451, -0.54226982, 0.61526217, -0.39364091, 1.6079713, 1.800822, 0.0276871])],
        # score: 0.55
        'qml_circuit_qiskit_05': [14, np.array([0.8147497, 0.31828516, -0.23239712, 2.1266553, 1.40275018, 1.15526225, -0.47383174, -0.32645496, 0.68316686, 0.37511029, 0.66308869, 0.03096657])],
    },
    'iris': {
        # score: 0.9666666666666667
        'qml_circuit_qiskit_01': [25, np.array([0.31248397, 0.92735327, 0.13046677, -0.89088838, -4.4078398, 2.77875535, 5.27365762, 0.86460951, -2.85045521, -1.45793349, -0.47640809, -0.21478062, -1.29118882, -3.11754105, -4.2504616, 0.57688293])],
        # score: 0.9333333333333333
        'qml_circuit_qiskit_02': [25, np.array([0.13919226, 0.18270388, -0.28463716, -0.78224311, -0.17047022, -1.07501037, 1.1740135, 0.77343931, 0.3569284, -0.13701128, -0.32215885, 0.15166619, 0.71878845, 1.02272243, 0.41251137, 0.82683124])],
        # score: 0.8666666666666667
        'qml_circuit_qiskit_03': [24, np.array([0.55024463, -0.46952006, 0.60047865, -1.15761329, -0.17066948, -0.47005236, 0.57806216, 2.56420208])],
        # score: 0.9
        'qml_circuit_qiskit_04': [25, np.array([-0.64309237, -0.88628174, -1.13229032, -1.00885081, -0.95295728, -1.38925037, -0.35017316, 0.1867368, -0.90036821, 2.03042545, -0.85411814, -1.94870286, 2.14712303, -1.87555317, 3.1071952, 0.37133862, -1.61484455, 1.81578953, -1.13208911, -1.06493484, 1.94750327, -0.6971807, -1.65207098, -1.18706223])],
        # score: 0.9
        'qml_circuit_qiskit_05': [24, np.array([0.36039556, -1.35962637, 1.33024093, 1.67312724, 0.70765699, -0.13429772, 2.2874656, 1.39319192, 0.03232704, -1.25577827, -0.67972942, -0.62564868, -0.06183411, 0.16963549, 0.67514865, 0.0778156, -0.22833464, 1.0114088, 0.33075759, 0.1540828, 0.73275214, 0.7780446, 0.76718374, -0.75604693])],
    },
    'rain': {
        # score: 0.69
        'qml_circuit_qiskit_01': [4, np.array([-0.35216563, 0.285661, 0.14918683, 0.58511709, 0.11441286, -0.34025229, -1.49345637, -0.08925404, -0.57156143, -2.81815222, 0.03153941, -1.07590283, 0.10737047, 2.65462923, -0.5969596, 0.72333055, 1.54074068, 0.00369869, 0.46322938, -3.29132411])],
        # score: 0.66
        'qml_circuit_qiskit_02': [0, np.array([1.73464493, -0.77824552, -0.16886394, 0.01263505, 1.39197308, 0.5415262, -0.35767633, 0.19935198, 0.87265109, 0.00798568, -0.54503854, 1.36287271, 0.5172351, 0.23975554, 2.45645713, 0.79317101, -0.01626538, -0.44737921, -0.87298365, -0.16685742])],
        # score: 0.735
        'qml_circuit_qiskit_03': [4, np.array([0.16411836, 2.98671275, 0.09481121, -0.0129218, -0.04169274, 0.22280165, 3.05116092, 0.06221335, -0.05709185, -0.02374078])],
        # score: 0.67
        'qml_circuit_qiskit_04': [9, np.array([0.59714171, 1.31089449, -0.1330296, 0.08308303, -0.64565276, 1.68006295, 1.06697633, 0.83871991, -0.52570691, 0.58046283, 1.043812, -0.35857379, 2.0149314, -0.23374026, 0.87850426, 1.16313644, -0.75355677, 0.66742569, 0.1145171, 0.88208179, -0.52967341, -0.9852449, 0.53615836, 0.51036888, 0.07129964, 1.22587247, -0.73622375, 2.32539072, 0.56144431, 0.95107114])],
        # score: 0.235
        'qml_circuit_qiskit_05': [4, np.array([-1.69914502, -0.86857062, 0.40001185, -0.27739806, 0.35454764, -0.10423128, 1.59336821, 2.31727754, -0.3040645, 1.94387474, -1.49472427, 2.14468062, 0.29430359, -0.8123898, -0.19200011, 0.24652593, 0.04042782, 0.63560901, 0.90708, -1.15026494, 1.69141958, 0.14819452, 0.25511455, 0.13440835, -0.47519516, 2.53326424, 1.44624421, 0.16184856, 2.29084047, 0.39975337])],
    },
    'vlds': {
        # score: 0.535
        'qml_circuit_qiskit_01': [44, np.array([0.79380451, -0.04836638, -0.21010617, -0.06395297, 0.16777932, -0.1703994, -0.48691591, -0.01836393, -0.33315392, -0.27573365, 1.54838226, 0.40033288, -0.09451476, 0.01371036, 0.86177265, 0.28633798, 0.29815943, 0.59411449, 0.59194236, 0.28157277])],
        # score: 0.68
        'qml_circuit_qiskit_02': [42, np.array([0.7575911, 0.18690081, 0.55793431, -0.35986827, -0.22653279, 0.20536795, -0.37958331, 0.9920072, -0.5415426, 0.35340794, -0.39336929, 1.00264781, -0.60742735, 0.20944543, 0.4286496, 0.05474495, 0.31583697, 0.05478755, 0.65019116, -0.24452015])],
        # score: 0.78
        'qml_circuit_qiskit_03': [46, np.array([-0.13835962, -0.00194082, 1.26005921, -0.01428553, 0.00537693, 0.22604437, -0.04197863, -0.15799963, 0.02365186, 0.06014628])],
        # score: 0.565
        'qml_circuit_qiskit_04': [41, np.array([1.53305539, 0.81849989, 1.49304288, 1.06100439, 1.21051202, -0.22396943, 0.40578617, -0.92353546, 0.70568243, 1.05344571, -0.00344706, 0.89764779, -0.60903121, -1.5936991, 2.08524417, -1.36194629, -0.08982061, 0.51153601, -0.59164933, 1.54055309, 0.80625441, -0.87921353, 1.05862963, 0.96744784, -0.35884513, 1.59069113, -0.39877665, -0.06608024, 0.73010087, 0.4816539])],
        # score: 0.465
        'qml_circuit_qiskit_05': [44, np.array([-0.66129984, 0.29756829, -0.35952493, -0.02995614, 2.03911878, 1.64194161, 1.5391168, 0.41279242, 3.03050651, -0.44285433, -0.6564261, 0.85649461, -2.38119571, -0.46220874, -1.42022374, -0.8650657, -0.02183638, 0.16680178, 0.32800685, -0.19055955, 1.57168741, -0.02251083, -3.09864086, -0.0093006, 0.99730819, 4.08984566, 2.83799715, 0.62608694, -2.32252594, 0.94112422])],
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
