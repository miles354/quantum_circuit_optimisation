from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization import circuit_drawer
from qiskit.visualization.bloch import Bloch
from math import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create the quantum circuit
qc = QuantumCircuit(4)
qc.h([0, 1, 2, 3])
qc.cx(0, 1)
qc.cy(1, 2)
qc.cx(3, 1)
qc.cz(0, 2)
qc.ccz(0, 1, 2)
qc.ccx(2, 1, 0)
qc.rx(pi / 3, 0)
qc.ry(pi / 4, 1)
qc.rz(pi / 5, 2)
qc.cz(0, 2)
qc.swap(1, 2)
qc.ry(pi / 2, 1)
qc.t(2)
qc.sdg(0)

# Draw and display the circuit diagram
circuit_fig = circuit_drawer(qc, output="mpl")
circuit_fig.tight_layout()
circuit_fig.savefig("TestQ_circuit_diagram.png")  # Save the circuit diagram
circuit_fig.show()  # Pops up the circuit diagram

# Simulate the final quantum state
state = Statevector.from_instruction(qc)

# Get reduced density matrices
reduced_dms = [partial_trace(state, [j for j in range(4) if j != i]) for i in range(4)]

# Bloch vector calculation
def get_bloch_components(dm):
    x = 2 * dm.data[0, 1].real
    y = 2 * dm.data[0, 1].imag
    z = dm.data[0, 0].real - dm.data[1, 1].real
    return [x, y, z]

# Create 2x2 grid of Bloch spheres
fig = plt.figure(figsize=(10, 10))
axes = [fig.add_subplot(2, 2, i + 1, projection='3d') for i in range(4)]
fig.suptitle("Bloch Spheres for All Qubits", fontsize=16)

for i, (dm, ax) in enumerate(zip(reduced_dms, axes)):
    b = Bloch(fig=fig, axes=ax)
    b.add_vectors(get_bloch_components(dm))
    b.render()
    ax.set_title(f"Qubit {i}")

fig.tight_layout()
fig.subplots_adjust(top=0.9)
fig.savefig("TestQ_bloch_spheres.png")  # Save the figure
plt.show()  # Pops up Bloch spheres
