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
single_qubit_gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 't', 's', 'sdg', 'u']
two_qubit_gates = ['cx', 'cz', 'swap']
three_qubit_gates = ['ccx']

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
def decompose_to_native(circ):
    native = QuantumCircuit(circ.num_qubits)

    for instr, qargs, _ in circ.data:
        gate = instr.name
        if len(qargs) == 1:
            qubit = circ.qubits.index(qargs[0])
        else:
            qubit = tuple(circ.qubits.index(q) for q in qargs)

# Gate Conditions
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



        elif gate == 'ccx':
            q1, q2, q3 = qubit
            # Only handle linear neighbours
            if abs(q1 - q2) == 1 and abs(q2 - q3) == 1:
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
