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
OPTIMIZER = {'COBYLA': {'settings': 'maxiter=1000, disp=False, rhobeg=1.0, tol=None, options=None'}}

# WEIGHTS
# add weights for best run (training score)
# Pretrained weights (best run)
pre_trained_weights = {
    'adhoc': {
        # score: 0.7
        'qml_circuit_qiskit_01': [21, np.array([0.02585034, 1.52917651, 0.78508577, 0.22745421, 0.84799742, 0.33558974, 0.32280995, 0.76783392, 0.44312809, 0.93748543, 0.21416306, 0.19550885])],
        # score: 0.65
        'qml_circuit_qiskit_02': [29, np.array([0.32500748, 1.11958844, 1.39582735, 1.21291758, 0.24623959, -0.59120644, -0.3171064, 1.38951459, 1.24151697, 0.57542269, -0.47844723, 0.321805])],
        'qml_circuit_qiskit_03': [21, np.array([1.16246464, 0.40220337, -0.28401591, 0.12021013, -0.0214482, -0.11472843])],   # score: 0.7
        # score: 0.75
        'qml_circuit_qiskit_04': [24, np.array([1.8236, 2.04374674, 0.311556, 1.32186649, 0.64870593, 0.91962146, 0.61950344, -0.45619532, 1.28120336, 1.50956981, 0.93232195, 0.66573179, 0.13704838, 0.97335602, 0.1522708, 1.72132845, -0.0250104, 0.65052769])],
        # score: 0.7
        'qml_circuit_qiskit_05': [24, np.array([1.1790398, 0.40215994, 0.74272357, 1.52170863, 0.54502014, 0.51243896, 0.15247088, 0.88710697, 0.31953563, 0.61216634, 0.39170489, 1.41673629, 1.91544544, 0.4786453, 1.1278806, 2.18386602, 0.11398436, 0.23509041])],
    },
    'custom': {
        'qml_circuit_qiskit_01': [2, np.array([1.20483351, 1.3255419, 1.50645121, -0.38416115, -0.08304048, 0.86508737, 0.84698098, 2.01431683])],  # score: 0.55
        'qml_circuit_qiskit_02': [0, np.array([2.14097159, 2.07793969, 0.02690911, 0.75743491, -0.37620219, 0.47547315, 0.94108111, 2.49860163])],  # score: 0.65
        'qml_circuit_qiskit_03': [0, np.array([-1.0482297, -0.50684417, 0.21260768, -0.50924422])],  # score: 0.7
        # score: 0.6
        'qml_circuit_qiskit_04': [3, np.array([2.07428387, 0.43678931, 1.84624908, 1.23988591, 0.0222432, 1.68876876, 0.37587889, -0.2985469, 1.65862628, -0.72486358, 0.77678815, 1.96554061])],
        # score: 0.6
        'qml_circuit_qiskit_05': [9, np.array([1.72386899, 0.34888992, 0.05186704, 1.73827596, -1.41298164, 2.1892478, -0.05137707, 0.63567324, 1.5326086, 1.96398656, 2.03796105, 0.12035197])],
    },
    'iris': {
        # score: 1.0
        'qml_circuit_qiskit_01': [11, np.array([-0.08878676, 1.87861488, 0.84076267, 0.63503758, 1.26375734, 0.91204138, 1.61981656, 1.43356223, 2.02899986, 0.34918227, 2.62725858, 1.68559733, 0.72171385, 0.76487067, 0.64294403, 0.39771845])],
        # score: 1.0
        'qml_circuit_qiskit_02': [13, np.array([1.6505642, 1.07817066, 0.08364105, 2.20938522, 1.88762234, 1.00999321, 0.59897588, 1.31620698, 0.96646928, 0.4403402, 0.43358576, 1.10542704, 0.73463015, 0.0082553, 0.11413614, -0.10799578])],
        'qml_circuit_qiskit_03': [15, np.array([-0.15753457, 0.26805707, 0.69257831, 0.4798, 0.18596267, -0.06557294, 0.87138214, 0.65073886])],  # score: 1.0
        'qml_circuit_qiskit_04': [16, np.array([0.77322901, 1.13422924, 0.74741021, 0.50546654, 1.24491481, 0.86503832, 1.06933113, 1.40622015, 0.34155752, 0.04716162, 0.66077993, 0.04680966, 0.19361304, 0.51026119, 0.02190681, -0.52477768, 1.9532735, 1.80298713, -0.99378354, 0.18125771, 0.6159572, 3.03382765, 0.10780192, 1.51091918])],  # score: 1.0
        'qml_circuit_qiskit_05': [19, np.array([0.96817428, 0.5151874, 0.52754989, 1.48341184, 1.64619938, -0.19649976, -0.45702734, -0.05114944, 0.68313361, 0.3539097, -0.41298016, 1.55052968, 0.65489799, 0.14798738, 0.53273701, 2.05688983, 1.29860083, 2.23851673, 1.50938736, 0.9644279, 0.86114805, 0.62901752, 1.80926673, 2.22838481])],  # score: 1.0
    },
    'rain': {
        'qml_circuit_qiskit_01': [36, np.array([0.30663938, 2.76419622, 0.4554716, 1.2496412, 1.26571089, 0.41547555, 0.47504539, 0.78224594, 1.60648197, 0.76513674, 1.45199027, 0.1822684, 0.27730361, 1.08534302, 0.16785077, 1.69037547, 0.39643197, 0.65216897, 1.23561759, 0.97191592])],  # score: 0.75
        'qml_circuit_qiskit_02': [33, np.array([1.57176076, 0.27461427, 0.59696507, 1.73520674, 1.59665016, 1.42493524, 1.45495876, 0.42525393, 1.80526177, 0.80804038, 0.73948961, 0.12225782, -0.34501249, 0.17570142, -0.28915023, 0.17360153, 0.61519149, 0.21350502, 0.18519903, 0.20619804])],  # score: 0.75
        'qml_circuit_qiskit_03': [34, np.array([1.88016092, 2.34756553, 0.47839067, 0.70481994, 1.55485277, 1.63716094, 1.72893907, 0.29759295, 0.50701317, 0.71671308])],  # score: 0.75
        'qml_circuit_qiskit_04': [37, np.array([0.08022492, 0.30189504, 0.1949895, 0.20595966, 0.19195542, 0.73546286, 0.70613785, 0.21962988, 0.99975793, 0.36131996, 0.08681141, 0.76744354, 0.00355733, 0.45888039, 0.29559936, 0.98933193, 0.89873325, 0.65731159, 0.93739232, 0.80731178, 0.45598988, 0.37201507, 0.7349411, 0.75653132, 0.58600257, 0.82538753, 0.42558662, 0.78936606, 0.33861478, 0.17862567])],  # score: 0.65
        'qml_circuit_qiskit_05': [30, np.array([0.70236301, 1.00587783, 1.08662985, -0.06534729, 0.44136834, 0.72463405, 0.36196362, 1.07811847, -0.2058468, 0.64655464, 0.99817445, 0.44652475, 1.51599695, -0.31288581, 0.6596019, 0.23888764, 0.76975462, -0.23904598, 0.25356014, 0.29973224, 0.13997708, -0.05902622, 0.5248941, 0.32220345, 0.84747804, 0.92668583, 0.49288282, 0.70059433, 0.33616921, 0.36882873])],  # score: 0.75
    },
    'vlds': {
        'qml_circuit_qiskit_01': [40, np.array([1.91145037, 0.23722635, 0.31589751, 0.49175521, 0.29685315, 1.56811679, 2.07994797, 0.70658685, -0.69956514, -0.43034621, 1.16801684, -0.5419077, 1.01480217, 2.23701747, 2.02579177, 0.28856903, 0.95369524, 0.09242241, 2.03319162, 0.28875226])],  # score: 0.9
        'qml_circuit_qiskit_02': [47, np.array([1.76992556, -0.12487622, 1.2987128, 1.8503471, 0.03582935, 1.77816365, 1.13678686, 0.00640911, 0.94285344, -0.66256498, 0.56987081, 0.12935358, 2.32416531, 0.70741374, 1.41551201, -0.50487558, 1.24553694, 0.42096935, 0.12233776, 1.03441896])],  # score: 0.85
        'qml_circuit_qiskit_03': [40, np.array([1.34741235, 1.58204877, -0.07919165, 0.05245437, 0.02348448, 1.14116055, 1.57971278, 0.23657233, 0.9312713, 0.03607076])],  # score: 0.85
        'qml_circuit_qiskit_04': [40, np.array([1.70303337, 0.67624628, 1.95564244, 0.66798392, 0.61111376, 1.09801684, 1.01479312, 1.41605231, -0.39848196, 0.30210288, -0.49926598, 2.13543494, 0.35915192, 1.60694814, 1.68593519, 1.34951632, 1.06576301, 0.04595834, 1.49555154, 0.18597882, 1.41856234, 0.76928627, 1.48935866, 1.79077439, -0.60489064, 1.49584767, 0.67011229, 0.11122875, -0.09917929, 0.6375882])],  # score: 0.8
        'qml_circuit_qiskit_05': [47, np.array([1.50227097e+00, -5.78682060e-02, -9.64484160e-02, 2.61805706e+00, 1.27473421e+00, 9.64687903e-01, 5.53495156e-01, 1.23471049e-01, -1.06715406e-02, 1.70877135e+00, 2.24966487e-01, 1.94608662e+00, 6.95618933e-02, 1.68720475e-01, 3.05249458e-01, 3.74652493e-01, -8.53799895e-02, -1.58074915e-03, 7.11027676e-01, 1.47837918e+00, 1.09855476e-01, -2.39375088e-01, -4.54593204e-02, 1.21251273e+00, 5.86733343e-01, 2.07617574e-01, 1.95124240e+00, 1.05070156e+00, 2.54143991e-01, 1.69553801e+00])],  # score: 0.85
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
