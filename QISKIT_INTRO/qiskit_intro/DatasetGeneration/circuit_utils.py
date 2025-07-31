# circuit_utils.py

import math
from qiskit import QuantumCircuit

# Depth Calculation Functions

# Naive depth calculation: counts 1 per gate, ignoring parallelism
def compute_naive_depth(gate_list):
    time_slots = {}
    current_time = 0

    for gate in gate_list:
        if gate == 'null':
            continue  # Skip placeholder or non-operational gates
        current_time += 1  # Simplistic assumption: each gate contributes 1 unit

    return current_time


# More accurate logical depth calculation: accounts for parallel gates
def compute_depth(circuit):
    """Compute circuit depth assuming gates can be parallel if on disjoint qubits."""
    depth = 0
    layers = []  # Each layer is a set of qubits used in that timestep

    for gate in circuit:
        if isinstance(gate, str) or gate.get("name") == "null":
            continue  # Skip invalid or placeholder gates

        qubits = tuple(sorted(gate.get("qubits", [])))
        placed = False

        for layer in layers:
            if all(q not in layer for q in qubits):
                layer.update(qubits)
                placed = True
                break

        if not placed:
            layers.append(set(qubits))
            depth += 1

    return depth


# Qiskit Conversion

def convert_to_qiskit_circuit(gate_list, num_qubits=6):
    qc = QuantumCircuit(num_qubits)
    for gate in gate_list:
        if gate == "null" or not isinstance(gate, dict):
            continue  # skip invalid entries

        name = gate["name"]
        qubits = gate["qubits"]
        params = gate.get("params", [])

        try:
            if name in ["rx", "ry", "rz"]:
                qc_method = getattr(qc, name)
                qc_method(params[0], qubits[0])
            elif name == "u":
                qc.u(params[0], params[1], params[2], qubits[0])
            elif name in ["x", "y", "z", "h", "s", "sdg", "t", "tdg"]:
                getattr(qc, name)(qubits[0])
            elif name == "cx":
                qc.cx(qubits[0], qubits[1])
            elif name == "cz":
                qc.cz(qubits[0], qubits[1])
            elif name == "swap":
                qc.swap(qubits[0], qubits[1])
            elif name == "ccx":
                qc.ccx(qubits[0], qubits[1], qubits[2])
        except Exception as e:
            print(f"Error adding gate {name} on qubits {qubits} with params {params}: {e}")

    return qc

def convert_qiskit_to_gate_list(qc):
    """
    Convert a Qiskit QuantumCircuit into a list of dictionaries
    representing gates with their name, qubits, and params.
    """
    gate_list = []
    for instr in qc.data:
        gate_list.append({
            "name": instr.operation.name,               # Gate name like 'cx', 'h', etc.
            "qubits": [qc.qubits.index(q) for q in instr.qubits],  # Qubit indices
            "params": instr.operation.params             # Parameters like rotation angles
        })
    return gate_list
