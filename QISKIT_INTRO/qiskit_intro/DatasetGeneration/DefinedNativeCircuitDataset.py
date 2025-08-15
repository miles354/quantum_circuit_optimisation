# Importing necessary libraries
import random
import json
from math import pi
import matplotlib.pyplot as plt

import os

# Get absolute path to the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the output folder
output_dir = os.path.join(script_dir, "..", "DatasetGeneration")

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

from qiskit.qasm3 import dumps as qasm3_dumps
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization.bloch import Bloch
from collections import Counter

# Parameters for circuit generation
num_qubits = 3
num_gates = 20
dataset_size = 1000

# Defining allowed gate types
single_qubit_gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 't', 's', 'sdg', 'u']
two_qubit_gates = ['cx', 'cz', 'swap']
three_qubit_gates = ['ccx']

ALL_GATES = single_qubit_gates + two_qubit_gates + three_qubit_gates

# Converts a circuit into a vector of gate counts
def get_gate_count_vector(qc, gate_list=ALL_GATES):
    counts = Counter([instr.operation.name for instr in qc.data])
    return [counts.get(g, 0) for g in gate_list]

# Computes Bloch vector components from a single-qubit density matrix
def get_bloch_components(dm):
    x = 2 * dm.data[0, 1].real
    y = 2 * dm.data[0, 1].imag
    z = dm.data[0, 0].real - dm.data[1, 1].real
    return [x, y, z]

# Applies SWAP gate using three CNOT gates (for decomposing SWAP into native gates)
def apply_swap_decomposed(circ, q1, q2):
    circ.cx(q1, q2)
    circ.cx(q2, q1)
    circ.cx(q1, q2)

# Applies a CX gate using nearest-neighbor SWAP routing if needed
def apply_nearest_neighbor_cx(circ, q1, q2):
    if abs(q1 - q2) == 1:
        circ.cx(q1, q2)
    else:
        # Generate SWAP path to bring qubits next to each other
        path = list(range(min(q1, q2), max(q1, q2) + 1))
        if q2 < q1:
            path = path[::-1]
        for i in range(len(path) - 1):
            apply_swap_decomposed(circ, path[i], path[i + 1])
        circ.cx(path[-1], path[-2])
        for i in reversed(range(len(path) - 1)):
            apply_swap_decomposed(circ, path[i], path[i + 1])

# Converts a circuit to one using only native gates
def decompose_to_native(circ):
    native = QuantumCircuit(circ.num_qubits)

    for instr, qargs, _ in circ.data:
        gate = instr.name
        if len(qargs) == 1:
            qubit = circ.qubits.index(qargs[0])
        else:
            qubit = tuple(circ.qubits.index(q) for q in qargs)

        # Single-qubit gate decompositions
        if gate == 'h':
            native.rz(pi/2, qubit)
            native.sx(qubit)
            native.rz(pi/2, qubit)

        elif gate == 'x':
            native.x(qubit)

        elif gate == 'y':
            native.rz(pi, qubit)
            native.sx(qubit)
            native.rz(pi, qubit)

        elif gate == 'z':
            native.rz(pi, qubit)

        elif gate == 'rx':
            theta = instr.params[0]
            native.rz(-pi/2, qubit)
            native.sx(qubit)
            native.rz(theta, qubit)
            native.sx(qubit)
            native.rz(pi/2, qubit)

        elif gate == 'ry':
            theta = instr.params[0]
            native.rz(pi/2, qubit)
            native.sx(qubit)
            native.rz(theta, qubit)
            native.sx(qubit)
            native.rz(-pi/2, qubit)

        elif gate == 'rz':
            theta = instr.params[0]
            native.rz(theta, qubit)

        elif gate == 't':
            native.rz(pi/4, qubit)

        elif gate == 'tdg':
            native.rz(-pi/4, qubit)

        elif gate == 's':
            native.rz(pi/2, qubit)

        elif gate == 'sdg':
            native.rz(-pi/2, qubit)

        elif gate == 'u':
            theta, phi, lam = instr.params
            native.rz(phi, qubit)
            native.sx(qubit)
            native.rz(theta, qubit)
            native.sx(qubit)
            native.rz(lam, qubit)

        # Two-qubit gate decompositions
        elif gate == 'cx':
            q1, q2 = qubit
            apply_nearest_neighbor_cx(native, q1, q2)

        elif gate == 'cz':
            q1, q2 = qubit
            native.rz(pi/2, qubit)
            native.sx(qubit)
            native.rz(pi/2, qubit)
            apply_nearest_neighbor_cx(native, q1, q2)
            native.rz(pi/2, qubit)
            native.sx(qubit)
            native.rz(pi/2, qubit)

        elif gate == 'swap':
            q1, q2 = qubit
            apply_nearest_neighbor_cx(native, q1, q2)

        # Three-qubit gate decomposition for CCX (Toffoli)
        elif gate == 'ccx':
            q1, q2, q3 = qubit
            if abs(q1 - q2) == 1 and abs(q2 - q3) == 1:  # Only handle linear connectivity
                native.rz(pi/2, q3)
                native.sx(q3)
                native.rz(pi/2, q3)
                apply_nearest_neighbor_cx(native, q2, q3)
                native.rz(-pi/4, q3)
                apply_nearest_neighbor_cx(native, q1, q3)
                native.rz(pi/4, q3)
                apply_nearest_neighbor_cx(native, q2, q3)
                native.rz(-pi/4, q3)
                apply_nearest_neighbor_cx(native, q1, q3)
                native.rz(pi/4, q2)
                native.rz(pi/4, q3)
                native.rz(pi/2, q3)
                native.sx(q3)
                native.rz(pi/2, q3)
                apply_nearest_neighbor_cx(native, q1, q2)
                native.rz(pi/4, q1)
                native.rz(-pi/4, q2)
                apply_nearest_neighbor_cx(native, q1, q2)

        else:
            raise ValueError(f"Unhandled gate type: {gate}")

    return native

# Generates a random quantum circuit with various gates
def generate_random_circuit(num_qubits=6, num_gates=20):
    qc = QuantumCircuit(num_qubits)
    for _ in range(num_gates):
        gate_type = random.choice(single_qubit_gates + two_qubit_gates + three_qubit_gates)

        if gate_type in single_qubit_gates:
            q = random.randint(0, num_qubits - 1)
            if gate_type in ['rx', 'ry', 'rz']:
                angle = random.choice([pi * n / 4 for n in range(1, 9)])
                getattr(qc, gate_type)(angle, q)
            elif gate_type == 'u':
                theta, phi, lam = [random.choice([pi * n/4 for n in range(1, 9)]) for _ in range(3)]
                qc.u(theta, phi, lam, q)
            else:
                getattr(qc, gate_type)(q)

        elif gate_type in two_qubit_gates:
            q1, q2 = random.sample(range(num_qubits), 2)
            getattr(qc, gate_type)(q1, q2)

        elif gate_type in three_qubit_gates:
            q1, q2, q3 = random.sample(range(num_qubits), 3)
            getattr(qc, gate_type)(q1, q2, q3)

    return qc

# Dataset generation loop
dataset = []

for i in range(dataset_size):
    qc = generate_random_circuit(num_qubits, num_gates)
    state = Statevector.from_instruction(qc)

    native_qc = decompose_to_native(qc)

    # Compute Bloch vectors for each qubit
    reduced_dms = [partial_trace(state, [j for j in range(num_qubits) if j != k]) for k in range(num_qubits)]
    bloch_vectors = [get_bloch_components(dm) for dm in reduced_dms]

    # Construct data dictionary for this sample
    data = {
        "id": i,
        "depth": qc.depth(),
        "gate_count": qc.size(),
        "gate_count_vector": get_gate_count_vector(qc),
        "native_depth": native_qc.depth(),
        "native_gate_count": native_qc.size(),
        "native_gate_count_vector": get_gate_count_vector(native_qc),
        "circuit_diagram": qc.draw(output="text").__str__(),
        "decomposed_diagram": native_qc.draw(output="text").__str__(),
        "statevector": [complex(val).real if abs(val.imag) < 1e-10 else complex(val) for val in state.data],
        "bloch_vectors": bloch_vectors
    }

    data["qasm"] = qasm3_dumps(qc)
    dataset.append(data)

# Save dataset to JSON file

output_path = os.path.join(output_dir, "quantum_circuit_dataset_with_native.json")

with open(output_path, "w") as f:
    json.dump(dataset, f, indent=2, default=str)

print(f"Generated and saved {dataset_size} circuits with native decompositions to '{output_path}'")
