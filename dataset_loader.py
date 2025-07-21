import json
import os
import sys
from qiskit import QuantumCircuit
from qiskit.qasm3 import loads

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_circuit_from_dataset(index=0, path="quantum_circuit_dataset_with_native.json"):
    """Loads one circuit (by index) from your dataset JSON and returns a QuantumCircuit."""
    with open(path, "r") as f:
        data = json.load(f)

    if index < 0 or index >= len(data):
        raise IndexError(f"Index {index} is out of bounds for dataset of size {len(data)}.")

    qasm_str = data[index]["qasm"]
    qc = loads(qasm_str)
    return qc
