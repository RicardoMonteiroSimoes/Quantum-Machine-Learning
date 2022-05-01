#!/usr/bin/env python3

import pythonlib.qcircuits as qc
import os
import pickle
import numpy as np
# Qiskit Imports
from qiskit import IBMQ, Aer, QuantumCircuit, transpile, execute, assemble
from qiskit.circuit import Parameter
from qiskit.tools.monitor import job_monitor
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy
# sklearn
from sklearn.metrics import accuracy_score

# Settings
# change dir into script dir
abspath = os.path.abspath(__file__)
SCRIPT_DIRECTORY = os.path.dirname(abspath)
os.chdir(SCRIPT_DIRECTORY)

# VARS
DATASET_FILE = SCRIPT_DIRECTORY + '/../datasets/datasets_10.data'  # 10 per dataset
# DATASET_FILE = SCRIPT_DIRECTORY + '/../datasets/datasets.data' # 13 per dataset
NUMBER_DATASETS = 5
NUMBER_RUNS = 10
NUMBER_SAMPLES = 100

load_dataset_args = (DATASET_FILE,
                     NUMBER_DATASETS,
                     NUMBER_RUNS,
                     NUMBER_SAMPLES)

# OUTPUT_SHAPE (number of classes)
OUTPUT_SHAPE = 2

# Quantum Settings
N_QUBITS = 5
# IBM provider
IBM_PROVIDER = 'ibm-q'

# load IBMid account settings (e.g. access token) from `$HOME/.qiskit/qiskitrc`
IBMQ.load_account()

# get provider
provider = IBMQ.get_provider(IBM_PROVIDER)
backend_system = 'ibmq_manila'

# Number of layers
N_LAYERS = os.getenv('QC_N_LAYERS', 2)

# Quantum circuits
quantum_circuits = [
    ('qml_circuit_qiskit_01', qc.qml_circuit_qiskit_01),
    ('qml_circuit_qiskit_02', qc.qml_circuit_qiskit_02),
    ('qml_circuit_qiskit_03', qc.qml_circuit_qiskit_03),
    ('qml_circuit_qiskit_04', qc.qml_circuit_qiskit_04),
    ('qml_circuit_qiskit_05', qc.qml_circuit_qiskit_05)
]

# Optimizer
OPTIMIZER = {'COBYLA': {'settings': 'maxiter=1000, disp=False, rhobeg=1.0, tol=None, options=None'}}

# WEIGHTS
# add weights for best run (training score)
# Pretrained weights (best run)
pre_trained_weights = {
    'adhoc': {
        'qml_circuit_qiskit_01': np.array([-0.10511535, 0.64954487, 1.93420391, -0.50899113, 0.46932607, -0.21376011, 0.36373938, 1.5038279, 1.73870895, 2.24519027, -0.6743587, 0.84247449]),
        'qml_circuit_qiskit_02': np.array([2.00009029,1.70324229,1.03741584,0.82586359,1.4709002,0.22414925,-0.09665123,0.80627877,0.2031737,-0.24568435,0.62750693,-0.04667787]),
        'qml_circuit_qiskit_03': np.array([0.41108058,0.90005491,-0.74320882,0.53926062,-0.16071711,0.05032031]),
        'qml_circuit_qiskit_04': np.array([]), # todo: data missing
        'qml_circuit_qiskit_05': np.array([1.69630567,0.5924622,1.86281391,1.23758323,0.70209068,1.82015981,0.12294481,1.34782367,0.04950074,0.36947841,0.36985917,0.69265725,0.38555297,0.05560714,0.59945472,0.84523958,0.38899252,0.55151542]),
    },
    'custom': {
        'qml_circuit_qiskit_01': np.array([2.93026589, -0.13799654, 2.58055967, -0.13396331, 1.40695619, 0.16534963, 0.62812168, -1.276996]),
        'qml_circuit_qiskit_02': np.array([2.04356937,0.92033486,2.49160599,-0.56749987,2.33979387,0.15833252,1.70969177,-1.31126183]),
        'qml_circuit_qiskit_03': np.array([0.69051246,3.43165677,2.61118366,2.00109993]),
        'qml_circuit_qiskit_04': np.array([1.82012461,1.09280073,-0.13518414,1.43318841,0.43484337,1.63410337,1.7844002,-0.26071661,0.87996905,-0.79705781,1.59088807,0.44245441]),
        'qml_circuit_qiskit_05': np.array([1.28226741,0.22664609,1.8938072,0.34150098,1.33806173,0.33751265,-0.80088941,0.61386194,-0.92430898,-0.32126651,2.18265344,0.37294541]),
    },
    'iris': {
        'qml_circuit_qiskit_01': np.array([-0.10046139,1.68123977,0.15152334,0.45247632,0.10197917,0.6504461,1.35793211,0.42761141,-0.21322559,1.13550443,0.86649677,1.99462524,0.58500611,0.02400503,0.34333735,0.2661118]),
        'qml_circuit_qiskit_02': np.array([0.96011588,0.86207021,1.77031538,0.40743356,2.08871263,0.91646923,2.2396041,1.47660373,0.93196385,-0.01123651,0.9367352,0.83459308,0.27513934,0.57823404,0.64456944,0.73852268]),
        'qml_circuit_qiskit_03': np.array([0.10082676,0.17084573,0.53243058,0.57066486,0.1742308,-0.10369546,0.8765528,0.73861743]),
        'qml_circuit_qiskit_04': np.array([1.58137422,0.72248275,0.03534374,1.76563398,-0.42431983,0.13113089,1.39355934,0.49517529,0.46527711,1.89986475,1.6074255,1.38766447,1.5552398,0.06116213,1.4885863,1.82020478,-0.02068819,-0.68024622,1.57925691,0.71074102,0.46163198,0.72210243,0.68679198,0.27574688]),
        'qml_circuit_qiskit_05': np.array([3.15351236,0.37677977,0.9059105,0.77720168,2.76204551,0.17893858,0.16944972,1.52719301,2.3724757,0.47122479,1.76537552,-0.01256516,0.0279082,0.31767879,1.16522945,1.37586915,-0.10332479,0.07521983,0.64680273,0.87113874,1.55563793,0.77950425,0.76605361,0.8021542]),
    },
    'rain': {
        'qml_circuit_qiskit_01': np.array([2.37008064,0.68371603,1.09644058,0.73453884,1.06139417,0.87515763,-0.47043454,2.05566729,1.18078587,1.04188606,1.57329505,-0.11715183,0.04183426,1.8965166,1.42781606,0.3313715,0.81781485,0.95841921,1.12254038,0.02979711]),
        'qml_circuit_qiskit_02': np.array([2.60377164,0.04004236,1.36788682,0.22173002,1.11003256,0.64908566,0.89697038,1.71636384,0.45850314,0.74611331,1.19393247,0.04159991,1.44644931,1.39069568,1.98101157,0.67165039,0.21933787,0.03229366,0.14853478,1.31871608]),
        'qml_circuit_qiskit_03': np.array([2.20350404,3.09893694,0.13794222,1.13093134,1.5410509,-0.01353714,1.80026362,0.40098341,1.6097892,1.54367705]),
        'qml_circuit_qiskit_04': np.array([-0.36846565,2.31730694,1.21659822,0.69522388,-0.45018251,-0.15753543,-0.22603733,0.33975813,1.68765864,0.43351013,0.46154453,0.51791605,0.42647284,1.08553303,2.25348024,0.32309926,-0.03903491,0.79629699,-1.12400119,-0.36019512,0.00894818,1.14000224,1.92212838,0.71248401,0.21443316,0.68145084,-0.65155723,0.31443528,1.26730596,0.98546681]),
        'qml_circuit_qiskit_05': np.array([0.63991306,0.39321718,1.28443569,-0.00806348,1.82428762,0.7242088,-0.32917209,0.44912371,-0.48498048,2.06094769,-0.1968633,0.32593043,0.18335531,0.94412749,-0.15885689,1.23182762,0.98774463,0.58544064,0.11515227,0.22795597,0.85019837,-0.27700023,1.84458643,0.54038327,0.72485927,2.25363484,0.24331255,-0.18635315,0.18185679,1.8663782]),
    },
    'vlds': {
        'qml_circuit_qiskit_01': np.array([2.50331517,0.48455175,1.69116164,0.70267755,0.57453885,0.23415703,1.6157566,0.67857785,0.3607229,1.31148922,0.83900531,0.58801218,1.6634368,0.06060186,1.40743482,1.47148634,1.1717598,0.58545273,1.57743834,1.41212693]),
        'qml_circuit_qiskit_02': np.array([1.62255023,1.01086926,2.29745519,1.71264788,2.14794341,0.83132625,2.24535488,0.69487866,1.95281066,1.99015459,0.3402451,0.43056126,0.53089931,0.16400514,0.72266315,0.0111672,0.55807857,0.30155666,-0.2159586,-0.13625248]),
        'qml_circuit_qiskit_03': np.array([0.98460495,1.5949544,0.23718724,-0.42449432,-0.03010883,0.21367228,1.45450993,1.61929435,1.39444474,0.015649]),
        'qml_circuit_qiskit_04': np.array([]), # todo: data missing
        'qml_circuit_qiskit_05': np.array([1.69992118,2.35485067,-0.04557659,1.77903031,0.17614534,0.82789606,1.23238106,0.12644933,-0.21651387,1.53658634,0.20887391,0.18405065,0.29590544,0.55031849,1.34660588,1.76353429,0.21439358,-0.07867524,1.18377182,1.20082001,0.10194546,0.42018341,0.05230939,1.57018328,1.90968125,0.19474203,0.10593794,1.04129346,-0.07416381,0.47178083]),
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
    # hlp.verify_datasets_integrity(data_sets, number_datasets=NUMBER_DATASETS,
    #                               number_samples=NUMBER_SAMPLES, number_runs=NUMBER_RUNS)
    print("done\n")
    return data_sets

# MAIN
if __name__ == '__main__':
    print("Running script in folder \"{}\"".format(SCRIPT_DIRECTORY))
    datasets = load_verify_datasets(load_dataset_args)

    scores = {}

    # Use filtered datasets (13) like: `for index, dataset in enumerate([datasets[i] for i in [1, 14, 27, 40, 53]]):`
    # Use filtered datasets (10) like: `for index, dataset in enumerate([datasets[i] for i in [1,11,21,31,41]]):`
    # for index, dataset in enumerate(datasets):
    for index, dataset in enumerate([datasets[i] for i in [1, 11, 21, 31, 41]]):
        (dataset_id, dataset_name, data) = dataset
        print("{}: {}".format(index, dataset[1]))
        (sample_train, sample_test, label_train, label_test) = data

        X = np.concatenate((sample_train, sample_test), axis=0)
        Y = np.concatenate((label_train, label_test), axis=0)
        np.subtract(Y, 1, out=Y, where=Y == 2)  # fix labels

        assert len(X) == len(Y), "features and labels not of equal length."

        dataset_name_id_key = '{}_{}'.format(dataset_name, dataset_id)

        try:
            if (pre_trained_weights[dataset_name]):

                scores[dataset_name_id_key] = {}

                for q_circuit in quantum_circuits:
                    (circuit_name, q_circ_builder) = q_circuit

                    try:
                        pre_trained_weights_current_dataset = pre_trained_weights[dataset_name][circuit_name]
                        print("found circ: {}".format(circuit_name))

                        print("weights:", pre_trained_weights_current_dataset)
                        n_wires = len(X[0])
                        print("n_wires:", n_wires)

                        # initialize classes array
                        calculated_classes = np.zeros(len(sample_train))

                        # build q circuit
                        quantum_circuit: QuantumCircuit = q_circ_builder(n_wires=n_wires, n_layers=N_LAYERS)

                        # break at (testing)
                        break_at_index = 2

                        # Loop over all features
                        for index, input_feature_arr in enumerate(sample_train):
                            print("input_features:", input_feature_arr)
                            params = np.concatenate((pre_trained_weights_current_dataset, input_feature_arr), axis=0)
                            circuit_with_params = quantum_circuit.bind_parameters(params)
                            #print(circuit_with_params.draw(vertical_compression='high', fold=-1, scale=0.5))

                            # determine least busy backend
                            backend = provider.get_backend('ibmq_qasm_simulator')  # test
                            # backend = provider.get_backend(backend_system)

                            job = execute(circuit_with_params, backend, shots=1024)
                            # alternative to execute (transpile and assemble the circuits yourself):
                            # mapped_circuit = transpile(circuit_with_data, backend=backend)
                            # qobj = assemble(mapped_circuit, backend=backend, shots=1024)
                            # job = backend.run(qobj)

                            # monitor job
                            job_monitor(job, interval=60)

                            result = job.result()
                            counts = result.get_counts()
                            # get max
                            max_score_register_value = max(counts, key=counts.get)
                            # convert to int
                            int_value_score = int(max_score_register_value, 2)
                            # determine class
                            calculated_classes[index] = parity(int_value_score)

                            # break out of for loop for testing
                            if index == break_at_index:  # 3 elements
                                break

                        # print("Y:", Y[:break_at_index+1])
                        # print("classes:", calculated_classes[:break_at_index+1])
                        # average
                        current_circuit_overall_score = accuracy_score(label_train[:break_at_index+1], calculated_classes[:break_at_index+1])
                        # print("accuracy_score ", current_circuit_overall_score)
                        # add scores to dict
                        scores[dataset_name_id_key][circuit_name] = current_circuit_overall_score

                    except:
                        print("Skipping circuit: {}".format(circuit_name))
                        pass

        except:
            print("Skipping dataset: {}".format(dataset_name))
            pass

    print(f"\nOptimizer: {OPTIMIZER}")
    print(f"\nscores: {scores}")
