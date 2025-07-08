import random
import json
from math import pi
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization.bloch import Bloch
from collections import Counter

# Parameters
num_qubits = 6
num_gates = 20
dataset_size = 1000

# Gate sets
single_qubit_gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 't', 's', 'sdg']
two_qubit_gates = ['cx', 'cz', 'swap']
three_qubit_gates = ['ccx', 'ccz']

ALL_GATES = single_qubit_gates + two_qubit_gates + three_qubit_gates

def get_gate_count_vector(qc, gate_list=ALL_GATES):
    counts = Counter([instr.operation.name for instr in qc.data])
    return [counts.get(g, 0) for g in gate_list]

# Bloch vector calculation
def get_bloch_components(dm):
    x = 2 * dm.data[0, 1].real
    y = 2 * dm.data[0, 1].imag
    z = dm.data[0, 0].real - dm.data[1, 1].real
    return [x, y, z]

# Helper for decomposing SWAP into native gates
def apply_swap_decomposed(circ, q1, q2):
    circ.cx(q1, q2)
    circ.cx(q2, q1)
    circ.cx(q1, q2)

# Helper for applying CX with nearest-neighbor SWAPs
def apply_nearest_neighbor_cx(circ, q1, q2):
    if abs(q1 - q2) == 1:
        circ.cx(q1, q2)
    else:
        path = list(range(min(q1, q2), max(q1, q2) + 1))
        if q2 < q1:
            path = path[::-1]
        for i in range(len(path) - 1):
            apply_swap_decomposed(circ, path[i], path[i + 1])
        circ.cx(path[-1], path[-2])
        for i in reversed(range(len(path) - 1)):
            apply_swap_decomposed(circ, path[i], path[i + 1])

# Native gate decomposition
def decompose_to_native(circuit):
    native = QuantumCircuit(circuit.num_qubits)

    for instr, qargs, _ in circuit.data:
        gate = instr.name
        qubits = [circuit.qubits.index(q) for q in qargs]

        if gate == 'h':
            native.rz(pi / 2, qubits[0])
            native.sx(qubits[0])
            native.rz(pi / 2, qubits[0])

        elif gate == 'x':
            native.x(qubits[0])

        elif gate == 'y':
            native.rz(pi, qubits[0])
            native.sx(qubits[0])
            native.rz(pi, qubits[0])

        elif gate == 'z':
            native.rz(pi, qubits[0])

        elif gate == 'rx':
            theta = instr.params[0]
            native.rz(-pi / 2, qubits[0])
            native.sx(qubits[0])
            native.rz(theta, qubits[0])
            native.sx(qubits[0])
            native.rz(pi / 2, qubits[0])

        elif gate == 'ry':
            theta = instr.params[0]
            native.rz(pi / 2, qubits[0])
            native.sx(qubits[0])
            native.rz(theta, qubits[0])
            native.sx(qubits[0])
            native.rz(-pi / 2, qubits[0])

        elif gate == 'rz':
            native.rz(instr.params[0], qubits[0])

        elif gate == 't':
            native.rz(pi / 4, qubits[0])

        elif gate == 's':
            native.rz(pi / 2, qubits[0])

        elif gate == 'sdg':
            native.rz(-pi / 2, qubits[0])

        elif gate == 'cx':
            apply_nearest_neighbor_cx(native, qubits[0], qubits[1])

        elif gate == 'cz':
            q1, q2 = qubits
            native.rz(pi / 2, q2)
            native.sx(q2)
            native.rz(pi / 2, q2)
            apply_nearest_neighbor_cx(native, q1, q2)
            native.rz(pi / 2, q2)
            native.sx(q2)
            native.rz(pi / 2, q2)

        elif gate == 'swap':
            apply_swap_decomposed(native, qubits[0], qubits[1])

        elif gate == 'ccx':
            native.ccx(*qubits)

        elif gate == 'ccz':
            native.h(qubits[2])
            native.ccx(qubits[0], qubits[1], qubits[2])
            native.h(qubits[2])

        else:
            raise ValueError(f"Unhandled gate type: {gate}")

    return native

# Generate a single random circuit
def generate_random_circuit(num_qubits=6, num_gates=20):
    qc = QuantumCircuit(num_qubits)
    for _ in range(num_gates):
        gate_type = random.choice(single_qubit_gates + two_qubit_gates + three_qubit_gates)

        if gate_type in single_qubit_gates:
            q = random.randint(0, num_qubits - 1)
            if gate_type in ['rx', 'ry', 'rz']:
                angle = random.choice([pi * n / 4 for n in range(1, 9)])
                getattr(qc, gate_type)(angle, q)
            else:
                getattr(qc, gate_type)(q)

        elif gate_type in two_qubit_gates:
            q1, q2 = random.sample(range(num_qubits), 2)
            getattr(qc, gate_type)(q1, q2)

        elif gate_type in three_qubit_gates:
            q1, q2, q3 = random.sample(range(num_qubits), 3)
            getattr(qc, gate_type)(q1, q2, q3)

    return qc

# Generate dataset
dataset = []

for i in range(dataset_size):
    qc = generate_random_circuit(num_qubits, num_gates)
    state = Statevector.from_instruction(qc)

    native_qc = decompose_to_native(qc)

    reduced_dms = [partial_trace(state, [j for j in range(num_qubits) if j != k]) for k in range(num_qubits)]
    bloch_vectors = [get_bloch_components(dm) for dm in reduced_dms]

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

    dataset.append(data)

# Save dataset
with open("quantum_circuit_dataset_with_native.json", "w") as f:
    json.dump(dataset, f, indent=2, default=str)

print(f"Generated and saved {dataset_size} circuits with native decompositions to 'quantum_circuit_dataset_with_native.json'")
