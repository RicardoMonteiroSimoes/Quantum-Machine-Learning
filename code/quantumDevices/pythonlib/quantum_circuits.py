from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
import pennylane as qml
from pennylane import QNode, Device


# Quantum Circuits

def qml_circuit_01(deviceOptions: dict) -> tuple[QNode, Device]:
    """
    Quantum Circuit
    """

    # create device
    q_device = qml.device(**deviceOptions)

    # define the circut
    @qml.qnode(q_device)
    def circuit(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(1))

    return circuit, q_device


# Todo: remove again
# Example functions ---------------------------
def testQuantumCircuit1(deviceName: str,
                        wireCount: int = 1,
                        shotCount: int = 1000  #  Defaults to 1000 if not specified
                        ) -> tuple[QNode, Device]:
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

    return circuit, q_device


def qiskitExampleCircuit() -> tuple[QuantumCircuit, Parameter]:
    """
    Qiskit Quantum Circuit
    """
    theta = Parameter('θ')

    qiskitCircuit = QuantumCircuit(2)
    qiskitCircuit.rz(theta, [0])
    qiskitCircuit.rx(theta, [0])
    qiskitCircuit.cx(0, 1)

    return qiskitCircuit, theta


def testQuantumCircuit_Qiskit_Import(deviceName: str,
                                     wireCount: int = 1,
                                     shotCount: int = 1000  #  Defaults to 1000 if not specified
                                     ) -> tuple[QNode, Device]:
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

    return quantum_circuit_with_loaded_subcircuit, q_device
