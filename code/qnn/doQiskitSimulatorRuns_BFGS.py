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
OPTIMIZER = {'BFGS': {'settings': 'maxfun=1000, maxiter=1500, ftol=2.220446049250313e-15, factr=None, iprint=- 1, epsilon=1e-08, eps=1e-08, options=None, max_evals_grouped=1'}}

# WEIGHTS
# add weights for best run (training score)
# Pretrained weights (best run)
pre_trained_weights = {
    'adhoc': {
        # score: 0.65
        'qml_circuit_qiskit_01': [29, np.array([-0.67676509, 0.80966908, -1.36759919, 0.5695726, 0.7030327, -0.236545, -1.23247623, 1.44919893, -0.55114697, 1.12625762, 0.92578021, 0.39400289])],
        # score: 0.8
        'qml_circuit_qiskit_02': [29, np.array([0.22486696, 1.45074931, -0.47274782, 0.74734906, 1.46126268, -0.45242768, -1.03386532, 0.83276932, 0.52778954, 0.64833728, 0.47232007, 0.75203736])],
        'qml_circuit_qiskit_03': [21, np.array([0.34459486, 0.19343597, -0.62165789, 0.758357, 0.32770606, -0.05242823])],  # score: 0.8
        # score: 0.65
        'qml_circuit_qiskit_04': [23, np.array([0.82534916, 0.18444683, 0.16967403, 0.89608465, 0.4495356, 0.16996163, 0.10867936, 0.82176128, 0.98727339, -0.13862841, -0.08760467, 0.41522879, 0.75959513, -0.20754496, -0.14547269, 0.63685242, 0.29325834, 0.8380724])],
        'qml_circuit_qiskit_05': [21, np.array([0.67160495, -0.17954073, 0.45548912, 1.75964512, 0.55773653, 0.61209966, 0.08603318, 1.51840393, 0.66702824, -0.30908555, 0.10789798, 0.27583343, 0.37788343, -0.11979149, 0.50108295, 0.57603051, 0.87361647, 0.09669828])],  # score: 0.7
    },
    'custom': {
        'qml_circuit_qiskit_01': [0, np.array([-0.28961589, 0.19412776, 0.24351081, -0.59174951, -0.01155586, 0.96160197, -0.23392438, -0.99123038])],  # score: 0.65
        'qml_circuit_qiskit_02': [0, np.array([1.00983105, -0.19946962, 2.23656885, -0.14028907, 0.77757411, -0.50717001, 2.07455164, 0.43085269])],  # score: 0.6
        'qml_circuit_qiskit_03': [0, np.array([-13.60511058, -11.58102802, -12.75580839, -20.79628212])],  # score: 0.7
        # score: 0.6
        'qml_circuit_qiskit_04': [9, np.array([1.79018399, 0.45444681, 1.9817005, -2.2808258, -1.27701733, -2.81865822, -1.24505164, -3.66848971, 3.68820263, 2.66634756, 0.58113549, 1.49614045])],
        # score: 0.65
        'qml_circuit_qiskit_05': [0, np.array([0.0055735, -0.60855208, -0.06949284, -0.89478624, 0.25556376, 0.18606759, 0.4699539, 0.09509044, 0.83572015, 0.26297069, 0.6278199, 0.07078532])],
    },
    'iris': {
        # score: 1.0
        'qml_circuit_qiskit_01': [10, np.array([0.63007896, 0.97717574, -0.67605354, 2.9087845, 1.16671038, -0.22231849, 2.43793139, 0.30019827, -0.31310324, 2.43588322, 3.62037149, 1.41438032, 0.07623225, 0.28745706, -0.51165033, -1.07519135])],
        # score: 1.0
        'qml_circuit_qiskit_02': [12, np.array([0.48823541, 0.06626862, 2.46392965, -2.55215487, 0.46960351, -0.64025023, 0.00424837, 0.45553385, 0.50758047, 3.03462558, 1.09750704, -0.03223732, 1.00780443, 0.25430094, -0.60921448, 0.98495181])],
        'qml_circuit_qiskit_03': [14, np.array([0.1463934, 0.00463647, 0.48386517, 0.38787112, 0.14162678, 0.01217745, 0.94443237, 0.69542193])],  # score: 1.0
        'qml_circuit_qiskit_04': [12, np.array([-0.49699649, -0.40673782, -0.06203446, 0.03543224, 0.67732041, 0.07797726, 0.50539894, 1.02326025, 0.48798487, 0.86611191, -0.13144747, 0.63653661, 0.45834449, 0.41936149, 0.1028484, -0.03715955, 0.1921535, 0.17144274, 0.3155288, -0.00762057, 0.71065673, 0.40624658, 0.96158074, 0.05635297])],  # score: 0.95
        'qml_circuit_qiskit_05': [19, np.array([-9.17786260e-04, 1.36131280e-01, 1.18745815e+00, 1.01017260e+00, 3.25068831e-01, 3.51091382e-01, 9.84828391e-02, 1.49832971e+00, 6.67455972e-02, 6.00782078e-01, 1.78421400e+00, 1.69664925e-01, -1.50554576e-02, 7.71958205e-02, 1.22896500e+00, 9.00396489e-01, -7.23083510e-02, 1.55678033e-01, 7.07008268e-01, 1.20292375e-01, 1.43714691e-01, 3.26943199e-02, 1.14232544e-01, 5.16846273e-01])],  # score: 1.0
    },
    'rain': {
        'qml_circuit_qiskit_01': [34, np.array([1.49332489, 0.15169767, 1.44885865, 0.54457327, 1.12645198, 0.19887098, 1.17842348, 0.37518084, 0.14991986, -0.63291942, 1.07454387, 1.02585167, 2.24519774, 0.06413764, 2.17860753, -0.22436047, -0.03875978, -0.44016993, 0.08312475, -0.11781965])],  # score: 0.8
        'qml_circuit_qiskit_02': [35, np.array([1.78809581, 0.17993664, 1.75285081, 0.97122935, 0.05653338, 0.06374533, 0.39713296, -0.12369827, 0.49029129, 1.63126227, 1.19629354, 0.44917355, 1.45134409, 0.67030977, 2.52564498, 0.20921753, 0.90438199, -0.29036039, -0.10097856, 1.70704367])],  # score: 0.8
        # score: 0.8
        'qml_circuit_qiskit_03': [32, np.array([2.02270331e-02, 5.23026193e-01, 6.56651266e-01, 4.00499472e-01, 2.92752755e-04, 1.28368416e-01, 4.49192969e-01, 6.19058098e-01, -3.02487967e-01, 3.99384828e-01])],
        'qml_circuit_qiskit_04': [35, np.array([0.54526333, 0.94500584, 0.51867171, 0.30559865, 0.57287479, 0.05617088, 0.33884339, 0.56694355, 0.51449641, 0.77214375, 0.00916866, 0.59774147, 0.61389347, 0.69320837, 0.69662608, 0.72783421, 0.76711934, 0.3199015, 0.37884511, 0.60111242, 0.93072907, 0.20529187, 0.44264853, 0.28403049, 0.08995844, 0.96706691, 0.63703986, 0.64889848, 0.1810667, 0.11100234])],  # score: 0.6
        'qml_circuit_qiskit_05': [38, np.array([1.58152766e-02, 6.87398960e-01, 2.29266598e-01, 1.84161706e-02, 2.13088835e-01, 1.59059470e-01, -2.24762043e-01, 1.61789850e+00, -2.82424748e-01, 1.30668394e-01, 2.47032330e-01, 1.38113010e+00, 1.26622902e+00, -9.10787699e-02, 1.83642961e-01, -3.55207818e-02, 9.56369236e-01, 2.22730012e-01, -3.48928064e-02, 1.06473335e+00, 2.09247679e-01, 3.63953707e-01, 1.63487043e+00, 2.01627586e-01, -2.10278876e-01, 1.69713661e-01, 6.37172707e-01, 7.37128896e-01, 1.90406196e-01, -1.42589828e-03])],  # score: 0.8
    },
    'vlds': {
        'qml_circuit_qiskit_01': [40, np.array([0.79156351, -0.01832138, 0.33585856, 1.39471514, 1.33979035, 0.51725805, 1.09636576, 0.1850349, 1.12757209, 0.41660908, 1.3546261, -0.00291323, 1.19305258, 1.30825681, 1.62737501, 0.22437225, 1.70526864, 0.12650293, -0.04966773, 0.00622383])],  # score: 0.8
        'qml_circuit_qiskit_02': [43, np.array([1.68379187, -0.71907848, 0.18078589, 1.55837858, 0.75539582, 1.30016341, 0.87943571, 0.3029028, 1.05328001, 0.3326402, 1.28334772, -0.40038995, -0.16024711, 1.12814266, 1.81453402, 0.05525445, 1.96855918, 0.57533197, 0.00591868, 0.00407186])],  # score: 0.7
        'qml_circuit_qiskit_03': [40, np.array([0.68686182, -1.24853362, 0.19478837, -0.88245256, 0.05477702, -0.09156873, 0.93341245, -0.0788358, -1.31934077, 0.10570258])],  # score: 0.8
        'qml_circuit_qiskit_04': [40, np.array([-3.01428071e-02, 5.02133869e-01, 8.63056141e-01, 1.20550064e+00, 3.24473112e-01, 7.34932118e-01, -1.91306326e-01, -4.00769855e-01, 1.04393496e+00, 1.08063181e+00, 3.44936642e-01, 1.06313790e+00, 3.91943450e-01, 8.23945026e-01, 9.95288336e-01, 1.06092592e+00, 3.03744519e-01, 6.95254737e-02, 1.11815583e+00, 6.48190282e-01, 4.92992523e-01, -2.17067351e-01, 3.65459724e-01, 5.88933663e-01, -5.61339050e-04, 9.50058172e-01, 6.70104259e-01, 3.08395196e-01, 5.53791576e-01, 2.88038853e-01])],  # score: 0.9
        'qml_circuit_qiskit_05': [49, np.array([1.99786609, -0.12321793, -0.14634229, 0.35944286, 1.00391277, -1.29685805, 0.40770322, -0.00663679, 0.52563722, 1.55583609, -0.09262518, 1.35570891, 0.61118777, 0.2286764, 0.90684511, 0.50097489, 2.39320443, -0.16173721, 2.60032507, 0.66385704, 0.00853777, 0.48973201, 0.1122839, 1.70507887, 0.79226394, 0.48565232, -0.06170531, 0.4648386, 0.40830173, 0.17413972])],  # score: 0.75
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
