import os
import sys
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import math

# Add parent directory for custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from GateRules.quantum_rules import apply_gate_identity, GATE_IDENTITIES
from DatasetGeneration.circuit_utils import convert_qiskit_to_gate_list, convert_to_qiskit_circuit

def create_circuit_for_identity(rule):
    """Creates a Qiskit circuit from a given gate identity rule."""
    qc = QuantumCircuit(6)

    name1 = rule["gate1"]
    name2 = rule["gate2"]
    condition = rule.get("condition", "")

    def apply_gate(circuit, name, qubits):
        if name == "cx":
            circuit.cx(qubits[0], qubits[1])
        elif name == "cz":
            circuit.cz(qubits[0], qubits[1])
        elif name == "swap":
            circuit.swap(qubits[0], qubits[1])
        elif name == "ccx":
            circuit.ccx(qubits[0], qubits[1], qubits[2])
        elif name == "rx":
            circuit.rx(math.pi / 4, qubits[0])
        elif name == "ry":
            circuit.ry(math.pi / 4, qubits[0])
        elif name == "rz":
            circuit.rz(math.pi / 4, qubits[0])
        elif name == "u":
            circuit.u(math.pi / 2, math.pi / 4, math.pi / 8, qubits[0])
        else:
            # Single-qubit gate with one argument
            getattr(circuit, name)(qubits[0])

    if "same_all_qubits" in condition:
        # Apply gate on multiple qubit sets
        apply_gate(qc, name1, [0, 1, 2])
        apply_gate(qc, name2, [0, 1, 2])
    elif "same_qubits" in condition:
        apply_gate(qc, name1, [0, 1])
        apply_gate(qc, name2, [0, 1])
    else:
        apply_gate(qc, name1, [0])
        apply_gate(qc, name2, [0])

    return qc

def check_equivalence(circ1, circ2):
    """Checks if two circuits are equivalent using fidelity."""
    state1 = Statevector.from_instruction(circ1)
    state2 = Statevector.from_instruction(circ2)
    fidelity = state1.inner(state2).real ** 2
    return fidelity >= 0.99999, fidelity

def simplify_using_identity_only(gate_list):
    """Simplify using only gate identity rules."""
    simplified = []
    i = 0
    while i < len(gate_list):
        g1 = gate_list[i]

        if i + 1 < len(gate_list):
            g2 = gate_list[i + 1]
            result = apply_gate_identity(g1, g2)
            if result:
                r1, _ = result
                if r1 != "null":
                    simplified.append(r1)
                i += 2
                continue

        simplified.append(g1)
        i += 1

    return simplified

def test_all_identity_rules():
    # Tests for all  identity rules for correctness
    for idx, rule in enumerate(GATE_IDENTITIES):
        print(f"\n[{idx+1}] Rule: {rule['gate1']} + {rule['gate2']} → {rule['replacement']}")

        try:
            original_circuit = create_circuit_for_identity(rule)
            print("Original Circuit:")
            print(original_circuit.draw())

            gate_list = convert_qiskit_to_gate_list(original_circuit)
            simplified_gate_list = simplify_using_identity_only(gate_list)
            simplified_circuit = convert_to_qiskit_circuit(simplified_gate_list, original_circuit.num_qubits)

            print("Simplified Circuit:")
            print(simplified_circuit.draw())

            equivalent, fidelity = check_equivalence(original_circuit, simplified_circuit)
            if equivalent:
                print(f"Circuits are equivalent (fidelity: {fidelity:.10f})")
            else:
                print(f"Circuits are NOT equivalent (fidelity: {fidelity:.10f})")

        except Exception as e:
            print(f"Error testing rule {rule}: {e}")

if __name__ == "__main__":
    test_all_identity_rules()
