import os
import json
from qiskit.qasm3 import loads
from qiskit import QuantumCircuit

def load_circuit_from_dataset(index=0, path=None):
    """
    Load a specific circuit from a dataset JSON file in QASM3 format.

    Args:
        index (int): Index of the circuit to load.
        path (str): Optional path to the dataset file. Defaults to local dataset file.

    Returns:
        QuantumCircuit: A Qiskit QuantumCircuit object.
    """
    if path is None:
        # Automatically construct the correct relative path
        path = os.path.join(os.path.dirname(__file__), "quantum_circuit_dataset_with_native.json")

    # Load JSON data from file
    with open(path, "r") as f:
        data = json.load(f)

    # Fallback if dataset is empty or index is invalid
    if index < 0 or index >= len(data):
        raise IndexError(f"Index {index} out of bounds for dataset of size {len(data)}")

    # Load QASM string and convert to QuantumCircuit
    qasm_str = data[index].get("qasm") or data[index].get("native_qasm")
    if not qasm_str:
        raise ValueError(f"No QASM data found at index {index} in dataset.")

    try:
        return loads(qasm_str)
    except Exception as e:
        raise RuntimeError(f"Failed to parse QASM3 for circuit at index {index}: {e}")
