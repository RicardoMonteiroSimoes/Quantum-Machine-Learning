import pythonlib.helper_functions as Helpers
import pythonlib.plot_functions as PlotHelpers
import pythonlib.quantum_circuits as QC
# ----
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np


wires = 2  # Number of qubits
# devices that use local simulators
q_deviceNames = [
    {"name": 'default.qubit', "wires": wires},  # Pennylane built-in
    {"name": 'default.mixed', "wires": wires},  # Pennylane built-in (noisy circuits)
    {"name": 'qiskit.aer', "wires": wires},  # Qiskit
    {"name": 'braket.local.qubit', "wires": wires},  # Amazon Braket
    {"name": 'cirq.simulator', "wires": wires},  # Google Cirq
    # {"name": 'strawberryfields.fock', "wires": wires, "cutoff_dim": 10},  # Strawberry Field's
    {"name": 'microsoft.QuantumSimulator', "wires": wires},  # Microsoft Q#
]

for simulatorArgs in q_deviceNames:
    qml_circuit_01, q_device = QC.qml_circuit_01(simulatorArgs)

    params = np.array([0.1, 0.2], requires_grad=True)
    result = qml_circuit_01(params)

    # print(qml.draw(qml_circuit_01)(params))
    # qml.drawer.use_style('black_white_dark')
    # fig, ax = qml.draw_mpl(qml_circuit_01)(params)
    # plt.show()

    # gradient
    dcircuit = qml.grad(qml_circuit_01)
    resultingGradient = dcircuit(params)

    print("\n[{}] gradient: ".format(simulatorArgs["name"]), resultingGradient)
