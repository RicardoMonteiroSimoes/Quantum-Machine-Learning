import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


# Quantum Circuits


def testQuantumCircuit1(deviceName,  # Todo: refactor whole func
                        wireCount=1,
                        shotCount=1000  #  Defaults to 1000 if not specified
                        ):
    """
    Test first Quantum Circuit
    """

    # create device
    q_device = qml.device(deviceName, wires=wireCount, shots=shotCount)

    # define the circut
    @ qml.qnode(q_device)
    def circuit(x):
        qml.RX(x, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

    return [circuit, q_device]


def qiskitExampleCircuit():
    """
    Qiskit Quantum Circuit
    """
    theta = Parameter('θ')

    qiskitCircuit = QuantumCircuit(2)
    qiskitCircuit.rz(theta, [0])
    qiskitCircuit.rx(theta, [0])
    qiskitCircuit.cx(0, 1)

    return [qiskitCircuit, theta]


def testQuantumCircuit_Qiskit_Import(deviceName,  # Todo: refactor whole func
                                     wireCount=1,
                                     shotCount=1000  #  Defaults to 1000 if not specified
                                     ):
    """
    Test imported Qiskit Quantum Circuit
    """

    # get qiskit circuit
    [qiskitCircuit, theta] = qiskitExampleCircuit()

    # create device
    q_device = qml.device(deviceName, wires=wireCount, shots=shotCount)

    # import qiskit ciruit
    @ qml.qnode(q_device)
    def quantum_circuit_with_loaded_subcircuit(x):
        qml.from_qiskit(qiskitCircuit)({theta: x})
        return qml.expval(qml.PauliZ(0))

    return [quantum_circuit_with_loaded_subcircuit, q_device]
