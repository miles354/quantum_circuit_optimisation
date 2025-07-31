# Import necessary libraries
import json
from qiskit.qasm3 import loads  # For loading circuits from QASM3 format
from qiskit import QuantumCircuit

# Convert a QuantumCircuit object to a list of gates with their respective qubit indices
def circuit_to_gate_list(qc: QuantumCircuit):
    """
    Extract gate names and the indices of qubits they act on from a QuantumCircuit.

    Returns:
        gate_list (list): A list of dictionaries, each representing a gate and its qubits.
    """
    gate_list = []
    for instr, qargs, _ in qc.data:
        gate_name = instr.name
        # Convert qubit objects to indices
        qubits = [qc.qubits.index(q) for q in qargs]
        gate_list.append({"gate": gate_name, "qubits": qubits})
    return gate_list

# Convert a list of circuits to JSON by extracting gate-level data
def convert_circuits_to_json(circuits, output_file="converted_rlgym_dataset.json"):
    """
    Convert a list of QuantumCircuit objects to a JSON dataset format containing gate lists.

    Args:
        circuits (list): List of QuantumCircuit objects.
        output_file (str): File name to save the output JSON data.
    """
    dataset = []
    for qc in circuits:
        gate_list = circuit_to_gate_list(qc)
        entry = {
            "circuit_diagram": gate_list,        # Gate-level representation
            "num_qubits": qc.num_qubits,         # Number of qubits in the circuit
            "depth": qc.depth(),                 # Depth of the circuit
            "size": qc.size()                    # Number of operations in the circuit
        }
        dataset.append(entry)

    # Save the dataset as a formatted JSON file
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} circuits to {output_file}")

# Load circuits from a JSON dataset that stores circuits in QASM3 format
def load_circuits_from_dataset(path):
    """
    Load QuantumCircuit objects from a JSON file with QASM3-formatted circuits.

    Args:
        path (str): Path to the JSON dataset.

    Returns:
        circuits (list): A list of loaded QuantumCircuit objects.
    """
    with open(path, "r") as f:
        raw_data = json.load(f)

    circuits = []
    for entry in raw_data:
        # Try to get QASM3 from 'qasm' or 'native_qasm' key
        qasm_str = entry.get("qasm") or entry.get("native_qasm")
        if not qasm_str:
            print(f"Skipping entry without QASM data: ID={entry.get('id', '?')}")
            continue

        try:
            # Parse QASM3 into a QuantumCircuit object
            qc = loads(qasm_str)
            circuits.append(qc)
        except Exception as e:
            print(f"Error loading circuit ID={entry.get('id', '?')}: {e}")

    return circuits

# Main script execution
if __name__ == "__main__":
    dataset_path = "quantum_circuit_dataset_with_native.json"  # Input dataset file
    circuits = load_circuits_from_dataset(dataset_path)

    if not circuits:
        print("No circuits loaded. Ensure the dataset includes valid 'qasm' entries.")
    else:
        # Convert the loaded circuits to JSON with gate-level info
        convert_circuits_to_json(circuits)
