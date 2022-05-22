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

# Optimizer: np\.array
OPTIMIZER = {'SPSA': {'settings': 'maxiter=100, blocking=False, allowed_increase=None, trust_region=False, learning_rate=None, perturbation=None, last_avg=1, resamplings=1, perturbation_dims=None, second_order=False, regularization=None, hessian_delay=0, lse_solver=None, initial_hessian=None, callback=None, termination_checker=None'}}

# WEIGHTS
# add weights for best run (training score)
# Pretrained weights (best run)
pre_trained_weights = {
    'adhoc': {
        # score: 0.7
        'qml_circuit_qiskit_01': [29, np.array([0.98804655, -1.39554406, -0.91980974, 0.23339586, 0.65289572, -3.43748878, 0.91937228, 0.22071598, 1.56803872, 0.73031619, 1.72846094, 0.31477971])],
        # score: 0.75
        'qml_circuit_qiskit_02': [29, np.array([3.69927649, -2.76625766, 2.89945554, 0.59728209, -4.334647, 1.57288613, 3.42508723, 0.04032684, -0.64066512, 2.00682983, 4.49112053, 1.01080538])],
        'qml_circuit_qiskit_03': [21, np.array([0.93403232, 0.7702825, -0.42154129, 0.29715533, 0.05632062, 0.19350282])],  # score: 0.7
        'qml_circuit_qiskit_04': [29, np.array([0.65414262, 4.66800793, -4.44020607, -3.07860264, -0.16340862, 2.78152696, -1.15375051, 2.18113124, 0.59909752, -4.36275593, 2.00083205, 0.87787627, -0.63280462, 2.15224311, 1.19189218, 3.95333651, -2.23166163, -2.42036503])],  # score: 0.7
        # score: 0.75
        'qml_circuit_qiskit_05': [21, np.array([1.21279799, -0.00212578, 0.60468641, 1.48209793, 0.6193126, -0.55725134, 0.95675373, 0.31604083, 0.17486197, 1.37835588, 0.05505657, 0.69651886, 0.64640494, 0.0764998, -0.7585041, 0.92599375, -0.128308, -0.66050157])],
    },
    'custom': {
        'qml_circuit_qiskit_01': [0, np.array([1.80144431, -1.02809938, 1.25430516, 4.12289639, -2.95571398, 0.61163972, 0.36343112, 0.6592655])],  # score: 0.55
        'qml_circuit_qiskit_02': [0, np.array([0.46689385, 2.20570733, 0.76605781, 2.95448773, -0.52120071, 0.62656292, -1.64944094, 1.21057502])],  # score: 0.65
        'qml_circuit_qiskit_03': [0, np.array([-1.53306125, -0.08828452, 0.62361089, -0.82414544])],  # score: 0.7
        # score: 0.55
        'qml_circuit_qiskit_04': [2, np.array([9.03111198, 1.48172423, 0.95693606, -5.64819526, 2.72030568, 6.44896255, -5.74209121, 0.27057275, -3.75976831, -1.65813488, -13.98125699, 3.95039636])],
        # score: 0.65
        'qml_circuit_qiskit_05': [0, np.array([1.32785098, -1.13750433, 0.0588378, 1.42936836, -0.78400412, 1.1612582, 1.7755458, -0.35956194, 0.89730961, 0.93741209, 0.04762136, -0.0691443])],
    },
    'iris': {

        # score: 1.0
        'qml_circuit_qiskit_01': [10, np.array([-0.87889351, -0.30041786, 0.86546259, -0.9117353, -0.37564247, 3.8440664, -1.00779181, -1.42943172, 3.27909094, 1.03230296, -1.0839181, 2.75398334, -1.00623777, -0.30469158, -0.52990776, -0.6750094])],

        # score: 1.0
        'qml_circuit_qiskit_02': [12, np.array([2.10473405, -0.98253247, 0.48655391, 0.35417131, 1.94189303, 1.29914247, 0.16655066, 1.87459103, 0.24634046, 0.78105082, 0.10534118, -0.18301564, 0.49406318, 0.05944511, -0.72515817, 0.64453577])],
        'qml_circuit_qiskit_03': [14, np.array([-0.0826751, -0.38223353, 1.20964197, 1.00071174, 0.36668422, 0.32540398, 0.11352896, 0.26610435])],  # score: 1.0
        'qml_circuit_qiskit_04': [13, np.array([12.27573652, -5.85972272, -12.4305078, -12.96922757, -1.17294784, -7.87821536, 9.61522867, -0.38056483, -2.29800089, 10.15306754, 1.30133996, 10.54600841, 1.71635533, -3.85339173, 14.78607215, 2.92693345, -7.30555488, -12.57644643, 1.36828835, -2.59016126, -0.88765226, 11.31592574, -5.24574311, -5.07857491])],  # score: 0.85
        'qml_circuit_qiskit_05': [19, np.array([0.07518027, 0.0108514, 1.83048079, -0.87159205, 0.91303977, 2.47264277, -0.33882333, 0.48908389, 0.02827497, 1.46752374, 1.02470599, 1.24983058, 2.02985675, 0.26645652, 0.06502935, -0.95685169, -0.45822034, 0.30312849, 1.68556779, -0.37800802, 0.29879001, -0.75866215, -0.9068052, 0.32623116])],  # score: 1.0
    },
    'rain': {
        'qml_circuit_qiskit_01': [35, np.array([-1.56527177, -3.17224382, 2.23746129, 4.21789741, 2.78592576, -4.58498262, 0.68717408, -0.11253571, -1.8721207, 1.72313388, -0.62973635, 3.18655122, -1.93938846, -2.22331976, 0.94127488, 2.48364258, -1.52245869, 0.01653174, 0.36554593, 2.62017405])],  # score: 0.8
        'qml_circuit_qiskit_02': [30, np.array([-1.69529623, 1.51517544, 1.55042402, -5.46358144, -1.6591387, -3.38737467, -0.08779879, -0.51542931, 0.44437084, 5.68296893, 4.14731844, 1.47462105, 3.97229758, -2.8183531, -3.16508956, -1.03095951, 5.60461661, 2.20425704, -3.71644275, 5.07673883])],  # score: 0.75
        'qml_circuit_qiskit_03': [36, np.array([-0.36297568, 0.6331478, 0.57024206, 1.25238265, 1.54808881, -0.04097649, 0.72796438, 0.59888476, 0.99491177, -1.42911894])],  # score: 0.8
        'qml_circuit_qiskit_04': [31, np.array([2.3801089, 1.0382973, -2.7742112, 5.22338017, -2.20009097, 0.36766552, -3.7857996, -2.13660455, 1.83597763, -2.41423766, -5.55384498, 2.19134021, 1.11995803, 3.11621925, 3.0942603, 3.18920392, 2.54149218, -1.73413051, 2.77419708, 7.63951632, 1.29834514, -2.91419216, -0.51970356, -1.5242197, -0.41588839, 0.08476527, 2.42968782, -1.50649558, -4.41563048, -2.58600619])],  # score: 0.7
        'qml_circuit_qiskit_05': [34, np.array([-0.0646279, 1.06716006, -0.04668721, -0.13244477, 1.44170434, 3.08800849, 0.52949886, 1.1786617, -0.22949709, 2.21112062, 0.67422179, 1.80529602, -1.53738991, 0.54288717, 0.42435727, -0.03902964, 0.10665449, 0.28613125, -0.5911414, 2.03994989, 0.23071878, 1.22312367, 1.76227901, 1.88382172, -1.32148007, -0.94797541, -0.13743032, 0.26820763, 1.8658781, 1.03160827])],  # score: 0.75
    },
    'vlds': {
        'qml_circuit_qiskit_01': [47, np.array([0.93876907, -2.57185059, -1.14700439, 0.52969709, 2.23797977, -1.12696913, 1.32269755, 0.85341603, -0.53101539, 2.59782481, 2.53388327, -0.42210816, 0.25329369, 1.74462342, 1.91650371, -0.62171153, 0.10422422, 2.4366311, 2.05971181, 3.05523662])],  # score: 0.8
        'qml_circuit_qiskit_02': [47, np.array([-1.67603698, -0.21438694, 2.70100693, 2.13342216, 0.23012696, 2.76870858, -0.08151312, 0.4194659, -1.41470863, 1.08854748, -4.17867718, -3.44335022, -0.71061965, -2.0134692, -0.60188681, 0.11207972, 1.90758097, 0.41322024, -0.53368182, 2.22321579])],  # score: 0.8
        'qml_circuit_qiskit_03': [40, np.array([0.24017437, 1.11942625, -0.89246317, 1.57536508, -0.0427855, 0.22846503, 1.76169585, -0.02841513, -0.63261733, 0.08476918])],  # score: 0.75
        'qml_circuit_qiskit_04': [40, np.array([2.28942538, 1.61065705, -4.53187404, -0.68757313, -2.69210848, -3.44309914, 0.52650379, -1.70501776, 0.32491475, 3.5992347, -4.60392043, 2.8184561, 0.62548959, -0.02762584, -1.76169642, 4.78241308, 6.41792155, 3.48428937, 0.85034821, 4.42842379, 4.44074905, 7.82371904, 0.25787427, 5.39126398, -3.18687301, 2.08349932, 5.7351019, -8.87520078, -6.48271947, -5.23444472])],  # score: 0.8
        'qml_circuit_qiskit_05': [40, np.array([1.72589313, 1.45112096, 0.90963098, 1.51050647, 0.8601673, 1.68002058, 1.51124701, 1.42511224, -0.23235317, 1.56239797, 0.10835655, 1.83986696, 0.60128226, 1.56100124, 0.59244187, -0.86049518, 1.53452343, -0.02747805, 1.29907115, 0.52458101, 0.04891656, -0.8469039, 0.02044887, -1.40543007, 0.54193764, 0.22449824, -0.05798178, 0.31822149, 0.68258526, 1.03342401])],  # score: 0.7
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
