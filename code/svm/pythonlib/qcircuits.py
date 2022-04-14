# pennylane
import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers, AngleEmbedding
from pennylane import broadcast
# qiskit
from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter
# import numpy as np

# Quantum Circuits


def qml_circuit(n_wires: int = 1,
                n_layers: int = 1,
                device_name: str = "default.qubit",
                diff_method: str = "parameter-shift",
                imprimitive=qml.ops.CNOT,
                expval_observable=qml.PauliZ(0)
                ):
    """
    Quantum Circuit (testing)
    """
    N_WIRES = n_wires
    N_LAYERS = n_layers

    wires = range(N_WIRES)

    # create the q device
    q_device = qml.device(device_name, wires=wires)

    @qml.qnode(q_device, diff_method=diff_method)
    def circuit(feature_vector, weights):
        """A variational quantum model."""
        # for i in list(wires):
        #     qml.Hadamard(wires=i)
        # embedding
        AngleEmbedding(features=feature_vector, wires=wires, rotation='Y')
        # trainable measurement
        StronglyEntanglingLayers(weights=weights, wires=wires, ranges=[1]*N_LAYERS, imprimitive=imprimitive)
        return qml.expval(expval_observable)

    def vc_classifier(weights, bias, feature_vector):
        return circuit(feature_vector, weights) + bias

    return circuit, vc_classifier, q_device


def qml_circuit_01(n_wires: int = 1,
                   device_name: str = "default.qubit",
                   diff_method: str = "parameter-shift",
                   expval_observable=qml.PauliZ(0)
                   ):
    """
    Quantum Circuit 1
    Figure 6.4 in PA

    n_layers is given trough weights

    Generate weights (Pennylane numpy):
    init_weights = np.random.randn(N_LAYERS, N_WIRES, 2, requires_grad=True)
    """
    N_WIRES = n_wires
    wires = range(N_WIRES)

    if N_WIRES <= 1:
        raise ValueError('At least two wires are required.')

    # create the q device
    q_device = qml.device(device_name, wires=wires)

    @qml.qnode(q_device, diff_method=diff_method)
    def circuit(feature_vector, weights):
        """A variational quantum model."""
        # template
        def q_template(param1, param2, wires):
            qml.RY(param1, wires=wires[0])
            qml.CRY(param2, wires=wires)
        # embedding
        AngleEmbedding(features=feature_vector, wires=wires, rotation='Y')
        # trainable measurement
        for parameters in weights:
            if N_WIRES == 2:
                qml.RY(parameters[0][0], wires=wires[0])
                qml.CRY(parameters[0][1], wires=[0, 1])
                qml.RY(parameters[1][0], wires=wires[1])
                qml.CRY(parameters[1][1], wires=[1, 0])
            else:
                broadcast(unitary=q_template, pattern='ring', wires=wires, parameters=parameters)

        return qml.expval(expval_observable)

    def vc_classifier(weights, bias, feature_vector):
        return circuit(feature_vector, weights) + bias

    return circuit, vc_classifier, q_device


def qml_circuit_02(n_wires: int = 1,
                   device_name: str = "default.qubit",
                   diff_method: str = "parameter-shift",
                   expval_observable=qml.PauliZ(0)
                   ):
    """
    Quantum Circuit 2
    Figure 6.5 in PA

    n_layers is given trough weights

    Generate weights (Pennylane numpy):
    init_weights = np.random.randn(N_LAYERS, 2, N_WIRES, requires_grad=True)
    """
    N_WIRES = n_wires
    wires = range(N_WIRES)

    if N_WIRES <= 1:
        raise ValueError('At least two wires are required.')

    # create the q device
    q_device = qml.device(device_name, wires=wires)

    @qml.qnode(q_device, diff_method=diff_method)
    def circuit(feature_vector, weights):
        """A variational quantum model."""
        # template
        def q_template(param, wires):
            qml.CRY(param, wires=wires)
        # embedding
        AngleEmbedding(features=feature_vector, wires=wires, rotation='Y')
        # trainable measurement
        for parameters in weights:
            if N_WIRES == 2:
                broadcast(unitary=qml.RY, pattern="single", wires=wires, parameters=parameters[0])
                qml.CRY(parameters[1][0], wires=[0, 1])
                qml.CRY(parameters[1][1], wires=[1, 0])
            else:
                broadcast(unitary=qml.RY, pattern="single", wires=wires, parameters=parameters[0])
                broadcast(unitary=q_template, pattern='ring', wires=wires, parameters=parameters[1])

        return qml.expval(expval_observable)

    def vc_classifier(weights, bias, feature_vector):
        return circuit(feature_vector, weights) + bias

    return circuit, vc_classifier, q_device


def qml_circuit_03(n_wires: int = 1,
                   device_name: str = "default.qubit",
                   diff_method: str = "parameter-shift",
                   expval_observable=qml.PauliZ(0)
                   ):
    """
    Quantum Circuit 3
    Figure 6.6 in PA

    n_layers is given trough weights

    Generate weights (Pennylane numpy):
    init_weights = init_weights = np.random.randn(N_LAYERS, N_WIRES, 1, requires_grad=True)
    """
    N_WIRES = n_wires
    wires = range(N_WIRES)

    if N_WIRES <= 1:
        raise ValueError('At least two wires are required.')

    # create the q device
    q_device = qml.device(device_name, wires=wires)

    @qml.qnode(q_device, diff_method=diff_method)
    def circuit(feature_vector, weights):
        """A variational quantum model."""
        # template
        def q_template(param, wires):
            qml.RY(param, wires=wires[0])
            qml.CZ(wires=wires)
        # embedding
        AngleEmbedding(features=feature_vector, wires=wires, rotation='Y')
        # trainable measurement
        for parameters in weights:
            if N_WIRES == 2:
                qml.RY(parameters[0][0], wires=wires[0])
                qml.CZ(wires=[0, 1])
                qml.RY(parameters[1][0], wires=wires[1])
                qml.CZ(wires=[1, 0])
            else:
                broadcast(unitary=q_template, pattern='ring', wires=wires, parameters=parameters)

        return qml.expval(expval_observable)

    def vc_classifier(weights, bias, feature_vector):
        return circuit(feature_vector, weights) + bias

    return circuit, vc_classifier, q_device


def qml_circuit_qiskit_01(n_wires=2, n_layers=1):
    """
    Quantum Circuit 1 (Qiskit)
    Figure 6.4 in PA
    """
    feature_map = QuantumCircuit(n_wires)
    ansatz = QuantumCircuit(n_wires)

    for i in range(n_wires):
        feature_map.ry(Parameter('i_{}'.format(str(i))), i)
    feature_map.barrier()
    for j in range(n_layers):
        for k in range(n_wires):
            ansatz.ry(Parameter('{}_w_{}'.format(str(j), str(k))), k)
            ansatz.cry(Parameter('{}_w2_{}'.format(str(j), str(k))), k, (k+1) % n_wires)
        if j != n_layers-1:
            ansatz.barrier()

    qc = QuantumCircuit(n_wires)
    qc.append(feature_map, range(n_wires))
    qc.append(ansatz, range(n_wires))
    return qc.decompose().copy()


def qml_circuit_qiskit_02(n_wires=2, n_layers=1):
    """
    Quantum Circuit 2 (Qiskit)
    Figure 6.5 in PA
    """
    feature_map = QuantumCircuit(n_wires)
    ansatz = QuantumCircuit(n_wires)

    for i in range(n_wires):
        feature_map.ry(Parameter('i_{}'.format(str(i))), i)
    feature_map.barrier()
    for j in range(n_layers):
        for k in range(n_wires):
            ansatz.ry(Parameter('{}_w_{}'.format(str(j), str(k))), k)
        for l in range(n_wires):
            ansatz.cry(Parameter('{}_w2_{}'.format(str(j), str(l))), l, (l+1) % n_wires)
        if j != n_layers-1:
            ansatz.barrier()

    qc = QuantumCircuit(n_wires)
    qc.append(feature_map, range(n_wires))
    qc.append(ansatz, range(n_wires))
    return qc.decompose().copy()


def qml_circuit_qiskit_03(n_wires=2, n_layers=1):
    """
    Quantum Circuit 3 (Qiskit)
    Figure 6.6 in PA
    """
    feature_map = QuantumCircuit(n_wires)
    ansatz = QuantumCircuit(n_wires)

    for i in range(n_wires):
        feature_map.ry(Parameter('i_{}'.format(str(i))), i)
    feature_map.barrier()

    for j in range(n_layers):
        for k in range(n_wires):
            ansatz.ry(Parameter('{}_w_{}'.format(str(j), str(k))), k)
            ansatz.cz(k, (k+1) % n_wires)
        if j != n_layers-1:
            ansatz.barrier()

    qc = QuantumCircuit(n_wires)
    qc.append(feature_map, range(n_wires))
    qc.append(ansatz, range(n_wires))
    return qc.decompose().copy()


def qml_circuit_qiskit_04(n_wires=2, n_layers=1):
    """
    Quantum Circuit 4 (Qiskit)
    No entanglement
    """
    feature_map = QuantumCircuit(n_wires)
    ansatz = QuantumCircuit(n_wires)

    for i in range(n_wires):
        feature_map.ry(Parameter('i_{}'.format(str(i))), i)
    feature_map.barrier()

    for j in range(n_layers):
        for k in range(n_wires):
            ansatz.ry(Parameter('{}_w_y{}'.format(str(j), str(k))), k)
            ansatz.rx(Parameter('{}_w_x{}'.format(str(j), str(k))), k)
            ansatz.rz(Parameter('{}_w_z{}'.format(str(j), str(k))), k)
        if j != n_layers-1:
            ansatz.barrier()

    qc = QuantumCircuit(n_wires)
    qc.append(feature_map, range(n_wires))
    qc.append(ansatz, range(n_wires))
    return qc.decompose().copy()


def qml_circuit_qiskit_05(n_wires=2, n_layers=1):
    """
    Quantum Circuit 5 (Qiskit)
    More rotation (Rx, Ry, Rz) with a CNOT (Strongly Entangled as ring)
    """
    feature_map = QuantumCircuit(n_wires)
    ansatz = QuantumCircuit(n_wires)

    for i in range(n_wires):
        feature_map.ry(Parameter('i_{}'.format(str(i))), i)
    feature_map.barrier()

    for j in range(n_layers):
        for k in range(n_wires):
            ansatz.rx(Parameter('{}_w_x{}'.format(str(j), str(k))), k)
            ansatz.ry(Parameter('{}_w_y{}'.format(str(j), str(k))), k)
            ansatz.rz(Parameter('{}_w_z{}'.format(str(j), str(k))), k)
            ansatz.cx(k, (k+1) % n_wires)
        if j != n_layers-1:
            ansatz.barrier()

    qc = QuantumCircuit(n_wires)
    qc.append(feature_map, range(n_wires))
    qc.append(ansatz, range(n_wires))
    return qc.decompose().copy()
