#!/usr/bin/env python3

import pythonlib.qcircuits as qc
import os
import pickle
import numpy as np
import pandas as pd
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

# IBM provider
IBM_PROVIDER = 'ibm-q'

# load IBMid account settings (e.g. access token) from `$HOME/.qiskit/qiskitrc`
IBMQ.load_account()

# get provider
provider: IBMQFactory = IBMQ.get_provider(IBM_PROVIDER)
# Hardware choose from: 'ibmq_manila' 'ibmq_quito' 'ibmq_belem' 'ibmq_lima'
# backend_system = 'ibmq_manila'
# backend_system = 'ibmq_quito'
# backend_system = 'ibmq_belem'
# backend_system = 'ibmq_lima'

# Simulators: aer_simulator (local), ibmq_qasm_simulator (managed online)
#backend_system = 'aer_simulator'
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
OPTIMIZER = {'ADAM AMSGRAD': {'settings': 'maxiter=1000, tol=1e-06, lr=0.001, beta_1=0.9, beta_2=0.99, noise_factor=1e-08, eps=1e-10, amsgrad=True, snapshot_dir=None'}}

# WEIGHTS
# add weights for best run (training score)
# Pretrained weights (best run)
pre_trained_weights = {
    'adhoc': {
        # score: 0.8
        'qml_circuit_qiskit_01': [29, np.array([0.07630828, 1.22970422, -0.0651395, 0.39183226, 0.21865134, -0.05041316, -0.18033653, 0.66927878, -0.71528421, 1.23625097, 0.33195928, 0.15167873])],
        # score: 0.75
        'qml_circuit_qiskit_02': [29, np.array([-0.18697519, 0.74060234, -0.86458712, 0.85351712, 0.82711374, 0.31018892, -0.73763127, 1.54543119, 0.17122193, 0.74773998, 0.86569667, 0.13238183])],
        'qml_circuit_qiskit_03': [21, np.array([0.56019659, 0.4574605, -0.26266296, 0.61407945, 0.26658388, -0.0055544])],  # score: 0.75
        'qml_circuit_qiskit_04': [29, np.array([0.00587867, -0.22275705, -0.02191617, 1.07745094, 0.21479886, 0.8934008, -0.07512348, 0.14711354, 1.50996056, 0.03859399, 0.23142698, 0.7926889, 0.60520445, -0.02627371, -0.03504661, 0.49976634, 0.24833128, 0.09365884])],  # score: 0.65
        # score: 0.75
        'qml_circuit_qiskit_05': [21, np.array([0.49021041, -0.13288564, 1.44367049, 0.64785304, 0.54981826, -0.45705166, 1.10392028, 1.43668358, 0.25462892, 1.2497005, 0.47799731, 0.70939592, 0.2389061, 0.57806016, 0.88704925, 0.43970023, -0.01146935, 0.2719283])],
    },
    'custom': {
        'qml_circuit_qiskit_01': [0, np.array([-0.29540564, 0.44252074, -0.6386595, -0.48140697, 0.68792073, 0.74256223, 0.30238779, -0.42532599])],  # score: 0.6
        'qml_circuit_qiskit_02': [0, np.array([0.62104385, -0.32025427, -0.64281016, 0.21215402, 0.78684627, -0.12439845, -0.55650062, -0.15502515])],  # score: 0.55
        'qml_circuit_qiskit_03': [0, np.array([-0.44742534, -0.22366666, -0.40683055, -0.21633652])],  # score: 0.55
        # score: 0.55
        'qml_circuit_qiskit_04': [2, np.array([0.08180276, -0.02626414, 0.14115286, -0.34184175, 0.21735755, 1.01737257, -0.06969894, 0.53363984, -0.05744692, -0.32770793, 0.18828458, 0.06141439])],
        # score: 0.55
        'qml_circuit_qiskit_05': [6, np.array([-0.24992258, 1.59572859, -0.66780293, 0.64863274, 0.01968169, 0.29459427, 1.09935585, 0.22656056, 1.29708391, 0.98843071, 0.59106936, 0.05584295])],
    },
    'iris': {
        # score: 1.0
        'qml_circuit_qiskit_01': [10, np.array([-0.17157065, 1.21376712, 0.37442282, 0.85114773, 0.44744956, 0.09449407, 0.89491815, 1.33229949, 0.22001866, 1.19538293, 0.7357702, 1.28090361, 0.32739243, 0.11608572, 0.84917121, -0.19104296])],
        # score: 1.0
        'qml_circuit_qiskit_02': [12, np.array([-0.28163945, 0.74731721, 0.809255, 1.15566414, 0.13731054, 0.50960219, 1.57455374, 0.65343114, 0.04491248, 0.97347177, 0.62106009, 1.08831996, 0.63596836, -0.14928698, 0.24740928, 0.10929216])],
        'qml_circuit_qiskit_03': [14, np.array([0.19377634, 0.08409543, 0.65098321, 0.62654044, 0.16234158, 0.03066866, 0.60682621, 0.60329686])],  # score: 1.0
        'qml_circuit_qiskit_04': [14, np.array([1.30649439, -0.23834209, 1.7076063, 0.18432451, 1.3910734, 0.70456464, 0.4608013, 1.20514393, -0.01405379, 0.54996844, -0.28173983, 0.4415028, -0.02648841, 0.39615597, 0.89538073, -0.47346727, 1.50160411, -0.23200959, 1.33789057, 1.10031745, 0.6254283, 0.58989277, 0.916514, 0.50278557])],  # score: 0.9
        'qml_circuit_qiskit_05': [19, np.array([-0.00764106, 0.02590556, 1.28305328, 1.51250146, 0.25780792, 0.31238047, 0.15702906, 1.03305577, 0.93373024, 0.72343838, 1.58591745, 0.81074249, -0.01850049, 0.0340039, 1.26218088, 0.35602141, 0.06065977, 0.15066103, 0.69731337, 0.29356309, 0.7918182, 0.04991875, 0.22639765, 0.45439976])],  # score: 1.0
    },
    'rain': {
        'qml_circuit_qiskit_01': [37, np.array([1.6006555, -0.52063128, 0.22237718, 1.214218, 0.56064172, -0.56062857, 0.7276212, 0.1175106, -0.09194491, 0.26110324, 1.05250745, -0.55027746, 0.49832499, 1.39560903, 0.82949494, -0.08123936, 0.5558871, 0.81058687, -0.33170012, 0.16320344])],  # score: 0.8
        'qml_circuit_qiskit_02': [38, np.array([1.17365198, -0.71695765, 1.03799049, 1.55626252, 0.20350365, -0.74065123, 0.96576627, 0.35418815, -0.09843194, 0.19967368, 1.54928543, -0.21588543, 0.53317565, 0.99169177, 0.744455, 0.04025327, 0.66133445, 0.22869659, -0.30941933, 0.42458391])],  # score: 0.8
        'qml_circuit_qiskit_03': [32, np.array([-0.50459531, 0.77548458, 0.51254783, 0.62893328, 1.13955088, 0.08330522, 0.45597417, 0.3779029, -0.08148654, -0.93365935])],  # score: 0.8
        'qml_circuit_qiskit_04': [38, np.array([0.94471139, 0.12870471, 0.26602351, 0.48952424, 0.82448731, 0.28889078, 0.22700606, 0.08056121, 0.4446035, 0.18182658, 0.07831126, 0.38240136, 0.43640601, 0.42448569, 0.74308533, 0.76823203, 0.77867447, 0.85608751, 0.82013088, 0.54107277, 0.28080937, 1.03169512, 0.14173921, 0.3068913, 0.8617741, 0.10081855, 0.0741105, 0.44748613, 0.915808, 0.85007854])],  # score: 0.7
        'qml_circuit_qiskit_05': [38, np.array([-0.1278208, 1.27995564, 0.78694184, 0.10734395, -0.01000398, 0.20581598, 0.11966281, 1.22783767, -0.07495036, -0.18484599, 0.46211395, 0.97646768, 0.38896331, -0.01395198, 0.86772206, 0.10510623, 1.38884578, 0.56421773, 0.02128318, 0.69293497, 0.06743371, 0.57515337, 0.40189619, -0.00679983, 0.73999786, 0.37701015, 0.15936884, 0.72069572, 0.01753544, 0.40228893])],  # score: 0.8
    },
    'vlds': {
        'qml_circuit_qiskit_01': [40, np.array([0.81050891, 0.26943497, 1.24241032, 0.76045058, 1.57912817, 1.0402882, 1.14506834, 0.30526441, -0.20797197, 1.26659779, 0.68355761, 0.11543989, 1.12914768, 1.18996756, 0.89012571, 0.60570678, 1.28810089, -0.07346169, 1.17721056, 0.77961777])],  # score: 0.9
        'qml_circuit_qiskit_02': [40, np.array([-0.29942724, 0.71323802, 0.410146, 1.4247302, 0.42519318, -0.47319767, -0.23467388, 0.44603317, -0.8255233, 0.83291345, 1.31518922, 0.75712157, 0.452076, 0.92477633, 0.24424934, 0.59320116, 0.06025624, -0.13556908, 1.55741419, 1.44987574])],  # score: 0.85
        'qml_circuit_qiskit_03': [40, np.array([0.62454685, 1.17627953, 0.26316273, 0.15744273, -0.6016923, 0.23144397, 1.4394619, 0.4571358, 0.61971488, 0.64656414])],  # score: 0.8
        'qml_circuit_qiskit_04': [40, np.array([-0.06709952, -0.18972765, -0.31151563, 0.10021059, 1.15409205, 0.2487466, 0.17245115, 0.45207868, 1.19093703, 0.7586928, 0.55485675, 0.34799005, 0.69545729, 1.43715169, 0.66423121, -0.08241436, 0.30535119, 0.3788677, 0.25690369, 0.17218572, 0.26972486, -0.34649492, 0.02665922, -0.13108526, 1.83436852, 0.75798299, 0.76288206, 0.61666075, 0.43746551, 0.51279946])],  # score: 0.8
        'qml_circuit_qiskit_05': [45, np.array([0.71471632, -0.03370578, 0.08789204, -0.32338525, 0.38889057, -0.11113375, 0.44049508, 0.44291743, 0.056872, 1.56587377, -0.08146354, 1.31456167, 1.35506299, 0.11216928, 0.81202918, 0.18610334, 0.88706438, -0.33384531, 0.88400802, 0.26152522, 0.01700219, 1.03426568, 0.44395348, 1.56275962, 0.65816577, 0.22445729, 0.73156726, 0.5619846, 0.51222395, 0.40807272])],  # score: 0.9
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
    # check if number of samples[train + test] is equal to number of samples as expected
    print("Verify datasets ...")
    for d in data_sets:
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
            (_, sample_test, _, label_test) = data
            np.subtract(label_test, 1, out=label_test, where=label_test == 2)  # fix labels (iris only)
            assert len(sample_test) == len(label_test) == 20, "test set: features and labels are not of equal length or not equal to 20."

            n_wires = len(sample_test[0])

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
                    job_manager = IBMQJobManager()
                    job_set = job_manager.run(transpiled_circuits_array, backend=backend, name=job_name, job_tags=job_tags)
                    results = job_set.results()

                for transpiled_circuit_index in range(len(transpiled_circuits_array)):
                    counts = results.get_counts(transpiled_circuit_index)
                    # get max
                    max_score_register_value = max(counts, key=counts.get)
                    # convert to int
                    int_value_score = int(max_score_register_value, 2)
                    # determine class
                    calculated_classes[transpiled_circuit_index] = parity(int_value_score)
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
