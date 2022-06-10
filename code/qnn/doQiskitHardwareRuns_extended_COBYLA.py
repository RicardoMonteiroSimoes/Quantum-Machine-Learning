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
OPTIMIZER = {'COBYLA': {'settings': 'maxiter=1000, disp=False, rhobeg=1.0, tol=None, options=None'}}

# WEIGHTS
# add weights for best run (training score)
# Pretrained weights (best run)
pre_trained_weights = {
    'adhoc': {
        # score: 0.535
        'qml_circuit_qiskit_01': [33, np.array([-0.38612387, 3.05525276, -0.17944086, -0.23799735, 0.284513, -0.78578391, 0.04205883, 2.72058025, -0.26714031, 0.40662069, -0.07035709, 1.05613948])],
        # score: 0.55
        'qml_circuit_qiskit_02': [34, np.array([0.20467362, 2.63834839, 0.22175381, 0.12431823, -0.11006674, -0.10349278, -0.03626498, 2.39275706, 0.31708962, -0.56539795, -0.01204076, 0.93112863])],
        # score: 0.525
        'qml_circuit_qiskit_03': [35, np.array([-0.08380193, 0.25133049, -0.05951762, -0.09626372, 0.27776987, -0.03412527])],
        # score: 0.515
        'qml_circuit_qiskit_04': [36, np.array([0.05156303, 0.14791167, 0.34318758, 0.53800155, -0.33427605, 1.01574162, 1.51745844, 2.13345487, 1.9645162, 0.61597735, -0.25063639, 1.06089233, 0.00470636, -0.30053896, 0.07304725, 1.85248695, 0.70928228, 0.28392333])],
        # score: 0.55
        'qml_circuit_qiskit_05': [35, np.array([1.34391782, 1.59197901, 0.37507065, 0.95771505, 1.52945814, 1.5937727, 1.04803344, 0.07502096, 0.68740529, -0.01157245, 1.5072313, 0.35411808, 0.01902578, 0.05003104, 1.61825509, 0.76575372, 0.23137109, 2.43971835])],
    },
    'custom': {
        # score: 0.56
        'qml_circuit_qiskit_01': [17, np.array([0.57750835, 0.6408067, 0.24454175, -0.93550066, 0.28127451, 0.09510245, -1.03201509, -0.00169481])],
        # score: 0.565
        'qml_circuit_qiskit_02': [10, np.array([-0.56383763, 1.14678396, -0.76349689, -0.05129182, 0.33393406, -0.03386664, -0.54609171, 0.04800648])],
        # score: 0.53
        'qml_circuit_qiskit_03': [17, np.array([-0.80812484, 0.25465456, 0.51413398, -0.44970941])],
        # score: 0.525
        'qml_circuit_qiskit_04': [17, np.array([0.93944311, -0.21589523, 1.43648656, 0.45570748, 1.67152493, 0.8079731, 1.41121179, 0.5101731, 0.35659193, -0.25759191, 1.42706438, 0.41836385])],
        # score: 0.555
        'qml_circuit_qiskit_05': [18, np.array([0.23840857, 2.39757316, 2.06576836, 2.00699123, 1.49000122, 0.10057187, 1.98611423, 0.83394669, 0.13332542, 0.664282, 1.3962579, 1.2888212])],
    },
    'iris': {
        # score: 0.9333333333333333
        'qml_circuit_qiskit_01': [25, np.array([-0.05537674, 2.2126556, 0.04030343, -0.51796335, 0.01738003, -0.76081762, 1.71026929, 0.65043509, 0.28947398, 1.877893, -0.15720936, -0.11179037, 0.51594613, 0.25786061, 0.31283495, 0.35649209])],
        # score: 0.9666666666666667
        'qml_circuit_qiskit_02': [22, np.array([-0.27504021, -0.14738353, 0.22113924, -1.1676391, 0.46113518, 0.13323107, 0.67461263, 1.25531381, 0.18093867, -0.55428842, -0.48519735, 0.48508604, 0.30615614, -0.10054477, 1.02953818, 0.36757454])],
        # score: 0.7666666666666667
        'qml_circuit_qiskit_03': [27, np.array([0.58074669, -0.19691314, 1.37843139, 0.43232157, 0.07001313, 0.04492007, 0.62774727, 0.62545145])],
        # score: 0.9666666666666667
        'qml_circuit_qiskit_04': [24, np.array([-0.28576404, 0.46198947, 0.65406177, 0.16495364, 0.95034519, 1.73776765, 0.27732945, 0.91223511, -0.12330869, 1.52382816, -0.31737082, 0.08515593, 0.07524084, 1.54524873, 0.0283728, -0.23845377, -0.68443555, 1.85889847, 1.17337151, 0.82952496, 1.1193291, 0.13256935, 1.59083876, -0.31660171])],
        # score: 0.8333333333333334
        'qml_circuit_qiskit_05': [20, np.array([0.58957857, 0.9010684, 1.8930639, 0.59232586, 1.24176498, 0.51702157, 1.50287486, 1.21241456, 0.14141455, 0.92810767, 0.04768582, -0.64122542, -0.16219389, 0.12358767, 0.59032352, 0.11946545, 0.70991683, 0.64330595, -0.19542986, 0.16884228, 0.58049483, 0.3181593, 1.33107992, 0.77727692])],
    },
    'rain': {
        # score: 0.705
        'qml_circuit_qiskit_01': [6, np.array([0.15431554, -0.2752421, -0.05854388, 0.43314508, 2.00812801, -0.8535586, -0.24699567, 0.76163289, 0.09103835, -0.11622995, 0.28511653, 0.37959699, -0.04404921, -0.28799092, 3.48130986, 1.1451743, -0.07162009, -0.94476651, -0.05909636, 0.07778689])],
        # score: 0.715
        'qml_circuit_qiskit_02': [9, np.array([1.31419423e-01, 1.11302823e-01, 6.95722726e-01, -4.54570898e-04, 2.57829531e+00, -9.98491302e-01, -3.53521771e-01, -2.72811969e-01, 8.27726395e-02, 9.91449959e-03, 4.75947877e-01, 4.59401126e-02, -5.50420429e-01, -2.52672766e-02, 3.05267461e+00, 1.26176093e+00, -3.25519730e-02, 9.04606541e-02, -1.62418669e-01, 5.00687807e-02])],
        # score: 0.73
        'qml_circuit_qiskit_03': [9, np.array([0.16541485, -0.1075523, -0.10690473, 0.00477448, 0.00523611, 0.32689179, -0.14389131, -0.02937387, -0.07471826, -0.03812691])],
        # score: 0.695
        'qml_circuit_qiskit_04': [6, np.array([-0.2106518, -0.37692959, 0.22077282, -0.48436677, 1.5550602, -0.23933358, 1.1359331, 0.47095115, 2.29754072, 1.04094104, 1.17379318, 1.58566572, 1.82996026, 1.64940427, 2.16784052, -0.27821726, 1.15038766, 0.68688914, 2.26427724, 1.47044578, 0.34460712, 0.46120286, -0.03765423, 0.47262832, 0.63949071, 0.44342405, 2.33032561, 0.98482495, 1.58927602, -0.3942225])],
        # score: 0.465
        'qml_circuit_qiskit_05': [6, np.array([1.10659313, 3.16528741, 0.05558599, 3.15064991, -0.0600045, 1.45812463, 0.00971226, 0.14577073, -0.10423356, 0.05861616, 0.84481832, -0.56188649, 0.82421219, 1.2631292, 1.83102797, -0.07090538, 0.0146216, 0.00763, 0.01656753, 0.04868361, 0.01507487, -0.02007562, -0.01073882, -0.02802629, -0.02779261, 0.69601576, 0.50339075, -0.03000868, 0.61190623, 0.89903497])],
    },
    'vlds': {
        # score: 0.905
        'qml_circuit_qiskit_01': [41, np.array([2.24166408e-01, 1.94620952e+00, 2.00866596e-01, -2.99446196e-01, 1.58599349e-01, -2.81527338e-01, -2.07230804e-01, -6.86282765e-01, -3.58912235e-02, 2.57792477e-01, -6.20492746e-02, 2.41384507e+00, 2.87666915e-01, 2.69937846e-01, -1.20895084e-03, 3.12338783e-01, 9.02881432e-02, 1.98690444e+00, -1.94238269e-01, -2.21282281e-01])],
        # score: 0.92
        'qml_circuit_qiskit_02': [49, np.array([-4.13492262e-02, 1.80931040e+00, 6.64546703e-02, 9.11240981e-02, 9.07868357e-01, 1.10903016e+00, 5.91488013e-02, 2.14169286e-01, -2.86978845e-01, 6.18409130e-02, -1.34980287e-02, 2.45851080e+00, -5.53612935e-02, -2.39894940e-01, -6.19949908e-01, -1.05635558e+00, -6.68148458e-02, 1.07892792e+00, 2.60141957e-01, 8.99914460e-04])],
        # score: 0.92
        'qml_circuit_qiskit_03': [46, np.array([-0.07746622, -0.07628466, 0.78567966, -0.00836927, 0.0175255, 0.09317275, -0.04661507, 0.48760881, 0.02128402, -0.01102599])],
        # score: 0.76
        'qml_circuit_qiskit_04': [46, np.array([1.28776851, 0.97996461, -0.63673449, 0.1664161, 0.13217015, 1.2391878, 0.88943081, 1.24786575, 1.38345756, 1.37069751, 2.01261252, 2.25499196, 1.5603022, 1.14412005, 1.82313295, 1.40615658, 1.17253826, 0.20033849, 2.06771505, 1.21991288, 0.76960917, 0.41367855, 0.65495865, 1.92237352, 0.76069468, 2.13304937, 0.41231118, 2.22282751, 1.25359857, 0.14158492])],
        # score: 0.465
        'qml_circuit_qiskit_05': [47, np.array([3.88754137e-02, 5.68182458e-02, -2.64369291e-02, -1.44897626e-02, -2.31853002e-02, 1.03036356e-01, 5.21043909e-02, 1.70988104e-01, 5.79013209e-03, 1.08604052e-02, 1.61567147e-01, -1.53545955e-01, 1.90332881e-01, 6.69516027e-01, 1.12093828e+00, -1.41791084e-02, 1.64554920e-02, -2.13999994e-02, 9.57820879e-03, 6.06540595e-02, 3.21994137e-02, 7.87900462e-03, 1.25507216e-03, -5.83750934e-02, -5.61841620e-02, 1.90428830e+00, 5.83930542e-02, 7.15409714e-01, 1.01261962e+00, 5.76131723e-02])],
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
