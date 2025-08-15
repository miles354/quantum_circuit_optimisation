from math import pi
import os
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

# --- Config ---
NUM_QUBITS = 1
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build circuit
qc = QuantumCircuit(NUM_QUBITS)

# Single-qubit gates
qc.ry(3 * pi/4, 0)

# Draw & save
fig = circuit_drawer(qc, output="mpl")
fig.tight_layout()

save_path = os.path.join(script_dir, "three_qubit_all_gates.png")
fig.savefig(save_path, dpi=300)
print(f"Circuit image saved to: {save_path}")

# Show (blocking)
plt.show(block=True)
