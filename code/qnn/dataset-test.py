import os
import numpy as np
import pickle


# change dir into script dir
abspath = os.path.abspath(__file__)
SCRIPT_DIRECTORY = os.path.dirname(abspath)
os.chdir(SCRIPT_DIRECTORY)

# VARS
# DATASET_FILE = SCRIPT_DIRECTORY + '/../datasets/datasets.data'
DATASET_FILE = SCRIPT_DIRECTORY + '/../datasets/datasets2.data'
NUMBER_RUNS = 10


def load_data(filename):
    with open(filename, 'rb') as filehandle:
        # read the data as binary data stream
        return pickle.load(filehandle)


data = load_data(DATASET_FILE)
# print(data)
for index, d in enumerate(data):
    #if index % NUMBER_RUNS == 0:
        (dataset_id, dataset_name, data) = d

        (sample_train, sample_test, label_train, label_test) = data

        N_WIRES = len(sample_train[0])  # feature count determines wires
        # count different values of targets (parity)
        OUTPUT_SHAPE = len(np.unique(label_train))

        print(f"""{dataset_id}: {dataset_name}
data
  - train: {sample_train[:1]} ({len(sample_train)})
  - test: {sample_test[:1]} ({len(sample_test)})
targets
  - train: {label_train[:10]} ({len(label_train)})
  - test: {label_test[:10]} ({len(label_test)})
train #: {len(sample_train)+len(sample_test)}
test #: {len(label_train)+len(label_test)}
features & wires #: {N_WIRES}
labels #: {OUTPUT_SHAPE}
""")

        # print(f"{dataset_id}: {dataset_name}")
