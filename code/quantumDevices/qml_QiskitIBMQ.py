import pythonlib.quantum_circuits as QC
import pythonlib.plot_functions as PlotHelpers
import pythonlib.helper_functions as Helpers
# ----
from pennylane import numpy as np
import pennylane as qml
from qiskit import IBMQ
import os
from dotenv import load_dotenv
load_dotenv()
# Create a `.env` file with the following content
# QISKIT_TOKEN=<TOKEN>

QISKIT_TOKEN = os.getenv('QISKIT_TOKEN')
IBMQ.save_account(QISKIT_TOKEN)
IBMQ.load_account()

wires = 2  # Number of qubits
# devices that use local simulators
q_deviceNames = [
    {
        "name": 'qiskit.ibmq',
        # "backend": 'ibmq_qasm_simulator',
        "backend": 'ibmq_lima',
        # "ibmqx_token": QISKIT_TOKEN,
        "wires": wires
    },
]

params = np.array([0.1, 0.2], requires_grad=True)

qml_circuit_01, q_device = QC.qml_circuit_01(q_deviceNames[0])

# print("capabilities backend:", q_device.capabilities()['backend'])

dcircuit = qml.grad(qml_circuit_01)

# Run the circuit
print("Result of circuit run on Qiskit:", qml_circuit_01(params))
# print result
print("Result of gradient calculation on Qiskit:", dcircuit(params))
