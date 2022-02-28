import pythonlib.helper_functions as Helpers
import pythonlib.plot_functions as PlotHelpers
import pythonlib.quantum_circuits as QC
import matplotlib.pyplot as plt
import numpy as np

# helper funcs example
Helpers.mytestfunc()

# plot funcs example
plt = PlotHelpers.contourPlot()
plt.show()

# Circuits examples
circuit1, device = QC.testQuantumCircuit1('default.qubit', 2)
print("result circuit1(0.5):")
print(circuit1(0.5))

circuit_with_import_from_qiskit, device = QC.testQuantumCircuit_Qiskit_Import('default.qubit', 2)
angle = np.pi/2
print("\nresult circuit_with_import_from_qiskit(angle):")
print(circuit_with_import_from_qiskit(angle))
