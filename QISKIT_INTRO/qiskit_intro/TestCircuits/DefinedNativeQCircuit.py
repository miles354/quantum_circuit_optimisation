import random
from math import pi
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization import circuit_drawer
from qiskit.visualization.bloch import Bloch

from collections import Counter

def get_gate_count_vector(qc):
    counts = Counter([instr.operation.name for instr in qc.data])
    return dict(counts)

# Parameters
num_qubits = 6
num_gates = 50

# Wider gate sets
single_qubit_gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 't', 'tdg', 's', 'sdg', 'u']
two_qubit_gates = ['cx', 'cz', 'swap']
three_qubit_gates = ['ccx']

# Create random circuit
qc = QuantumCircuit(num_qubits)
for _ in range(num_gates):
    gate_type = random.choice(single_qubit_gates + two_qubit_gates + three_qubit_gates)

    if gate_type in single_qubit_gates:
        q = random.randint(0, num_qubits - 1)
        if gate_type in ['rx', 'ry', 'rz']:
            angle = random.choice([pi * n/4 for n in range(1, 9)])
            getattr(qc, gate_type)(angle, q)
        elif gate_type == 'u':
            theta, phi, lam = [random.choice([pi * n/4 for n in range(1, 9)]) for _ in range(3)]
            qc.u(theta, phi, lam, q)
        else:
            getattr(qc, gate_type)(q)

    elif gate_type in two_qubit_gates and num_qubits >= 2:
        q1, q2 = random.sample(range(num_qubits), 2)
        getattr(qc, gate_type)(q1, q2)

    elif gate_type in three_qubit_gates and num_qubits >= 3:
        # To avoid ValueError later, force linear neighbours for ccx
        q = random.randint(1, num_qubits - 2)
        q1, q2, q3 = q-1, q, q+1
        qc.ccx(q1, q2, q3)

print("Original circuit:")
circuit_img_orig = circuit_drawer(qc, output="mpl")
circuit_img_orig.tight_layout()
circuit_img_orig.show()
circuit_img_orig.savefig("Original_circuit_diagram.png")

# Manual helper to fully decompose SWAP into 3 CX gates
def apply_swap_decomposed(circ, q1, q2):
    circ.cx(q1, q2)
    circ.cx(q2, q1)
    circ.cx(q1, q2)

# Helper to implement CX between arbitrary qubits with nearest neighbour swaps
def apply_nearest_neighbor_cx(circ, q1, q2):
    if abs(q1 - q2) == 1:
        circ.cx(q1, q2)
    else:
        path = list(range(min(q1, q2), max(q1, q2) + 1))
        if q2 < q1:
            path = path[::-1]
        # Swap qubit q1 to neighbour of q2
        for i in range(len(path) - 1):
            apply_swap_decomposed(circ, path[i], path[i+1])
        circ.cx(path[-1], path[-2])
        # Swap back
        for i in reversed(range(len(path) - 1)):
            apply_swap_decomposed(circ, path[i], path[i+1])

def decompose_to_native_fully(circ):
    native = QuantumCircuit(circ.num_qubits)

    for instr, qargs, _ in circ.data:
        gate = instr.name
        if len(qargs) == 1:
            qubit = circ.qubits.index(qargs[0])
        else:
            qubit = tuple(circ.qubits.index(q) for q in qargs)

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
            apply_swap_decomposed(native, q1, q2)

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
                raise ValueError("CCX must be on linear nearest-neighbour qubits")

        else:
            raise ValueError(f"Gate '{gate}' not handled in decomposition")

    return native

native_qc = decompose_to_native_fully(qc)

print("Decomposed circuit (fully native, manual swaps):")
circuit_img_native = circuit_drawer(native_qc, output="mpl")
circuit_img_native.tight_layout()
circuit_img_native.show()
circuit_img_native.savefig("DefinedNative_circuit_diagram.png")


# --- Bloch spheres visualisation (original circuit) ---

state = Statevector.from_instruction(qc)
reduced_dms = [partial_trace(state, [j for j in range(num_qubits) if j != i]) for i in range(num_qubits)]

def get_bloch_components(dm):
    x = 2 * dm.data[0, 1].real
    y = 2 * dm.data[0, 1].imag
    z = dm.data[0, 0].real - dm.data[1, 1].real
    return [x, y, z]

cols = 3
rows = (num_qubits + cols - 1) // cols
fig = plt.figure(figsize=(4 * cols, 4 * rows))
fig.suptitle("Bloch Spheres for All Qubits", fontsize=16)
axes = [fig.add_subplot(rows, cols, i + 1, projection='3d') for i in range(num_qubits)]

for i, (dm, ax) in enumerate(zip(reduced_dms, axes)):
    b = Bloch(fig=fig, axes=ax)
    b.add_vectors(get_bloch_components(dm))
    b.render()
    ax.set_title(f"Qubit {i}")

fig.tight_layout()
fig.subplots_adjust(top=0.9)
fig.savefig("Q_bloch_spheres.png")
plt.show()


print("\nOriginal circuit statistics:")
print(f"Depth: {qc.depth()}")
print(f"Total gates: {qc.size()}")
print("Gate counts by type:")
for gate, count in get_gate_count_vector(qc).items():
    print(f"  {gate}: {count}")

print("\nDecomposed (native) circuit statistics:")
print(f"Depth: {native_qc.depth()}")
print(f"Total gates: {native_qc.size()}")
print("Gate counts by type:")
for gate, count in get_gate_count_vector(native_qc).items():
    print(f"  {gate}: {count}")

print(f"\nQubit count: {num_qubits}")
