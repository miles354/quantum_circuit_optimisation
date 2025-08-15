import random
from math import pi
import matplotlib.pyplot as plt
import os

# Directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization import circuit_drawer
from qiskit.visualization.bloch import Bloch

# Parameters
num_qubits = random.randint(5, 7)
num_gates = random.randint(20, 40)

# Gate options
single_qubit_gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 't', 's', 'sdg']
two_qubit_gates = ['cx', 'cz', 'swap']
three_qubit_gates = ['ccx', 'ccz']

# Create random circuit
qc = QuantumCircuit(num_qubits)
for i in range(num_gates):
    gate_type = random.choice(single_qubit_gates + two_qubit_gates + three_qubit_gates)

    if gate_type in single_qubit_gates:
        q = random.randint(0, num_qubits - 1)
        if gate_type in ['rx', 'ry', 'rz']:
            angle = random.choice([pi * 1/4, pi * 1/2, pi * 3/4, pi, pi * 5/4, pi * 3/2, pi * 7/4, pi * 2])
            getattr(qc, gate_type)(angle, q)
        else:
            getattr(qc, gate_type)(q)

    elif gate_type in two_qubit_gates and num_qubits >= 2:
        q1, q2 = random.sample(range(num_qubits), 2)
        getattr(qc, gate_type)(q1, q2)

    elif gate_type in three_qubit_gates and num_qubits >= 3:
        q1, q2, q3 = random.sample(range(num_qubits), 3)
        getattr(qc, gate_type)(q1, q2, q3)

# Draw and display the circuit diagram
circuit_img = circuit_drawer(qc, output="mpl")
circuit_img.tight_layout()
circuit_path = os.path.join(script_dir, "RandomQ_circuit_diagram.png")
circuit_img.savefig(circuit_path)
circuit_img.show()

# Simulate final state
state = Statevector.from_instruction(qc)

# Reduced density matrices for each qubit
reduced_dms = [partial_trace(state, [j for j in range(num_qubits) if j != i]) for i in range(num_qubits)]

# Bloch vector calculation
def get_bloch_components(dm):
    x = 2 * dm.data[0, 1].real
    y = 2 * dm.data[0, 1].imag
    z = dm.data[0, 0].real - dm.data[1, 1].real
    return [x, y, z]

# Save Bloch spheres for all qubits in a grid
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

# fig.tight_layout()  # Avoid for 3D plots
fig.subplots_adjust(top=0.9)
bloch_path = os.path.join(script_dir, "RandomQ_bloch_spheres.png")
fig.savefig(bloch_path)
plt.show()


# Print circuit info
depth = qc.depth()
print("Circuit depth:", depth)
print("Gate count:", num_gates)
print("Qubit count:", num_qubits)
