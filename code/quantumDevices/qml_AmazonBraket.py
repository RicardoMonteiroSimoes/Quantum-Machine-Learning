import pythonlib.helper_functions as Helpers
import pythonlib.plot_functions as PlotHelpers
import pythonlib.quantum_circuits as QC
# ----
from braket.aws import AwsSession
import boto3
# import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np


# Set AWS Profile session [REQUIRED]
boto_sess = boto3.Session(profile_name='private_phuber', region_name='eu-west-2')
# S3 Bucket (must exist) [REQUIRED]
S3_Bucket = ('amazon-braket-34bf4c0b11c1', 'BA-testing-01')
# Device ARN [REQUIRED]
# See: https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html
device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"

# Initialize Braket session with Boto3 Session credentials
aws_session = AwsSession(boto_session=boto_sess)

wires = 2  # Number of qubits
# devices that use local simulators
q_deviceNames = [
    {
        "name": 'braket.aws.qubit',
        "aws_session": aws_session,
        "device_arn": device_arn,
        "s3_destination_folder": S3_Bucket,
        "wires": wires
    },
]

params = np.array([0.1, 0.2], requires_grad=True)

qml_circuit_01, q_device = QC.qml_circuit_01(q_deviceNames[0])

dcircuit = qml.grad(qml_circuit_01)

# Run the circuit
print("Result of circuit run on SV1:", qml_circuit_01(params))
# print result
print("Result of gradient calculation on SV1:", dcircuit(params))
