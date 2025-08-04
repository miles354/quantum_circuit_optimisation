import os
import sys
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector # Import necessary modules

# Add parent directory for custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from GateRules.quantum_rules import apply_conjugation_rule, CONJUGATION_RULES
from DatasetGeneration.circuit_utils import convert_qiskit_to_gate_list, convert_to_qiskit_circuit

def create_circuit_for_conjugation(rule):
    # Creates a Qiskit circuit from a given conjugation rule.
    qc = QuantumCircuit(6)

    name1 = rule["gate1"]
    name2 = rule["gate2"]
    name3 = rule["gate3"]
    qubit = 0  # default to qubit 0
    condition = rule.get("condition", "")

    if "same_all_qubits" in condition:
        qc.__getattribute__(name1)(0)
        qc.__getattribute__(name2)(0)
        qc.__getattribute__(name3)(0)
        qc.__getattribute__(name1)(1)
        qc.__getattribute__(name2)(1)
        qc.__getattribute__(name3)(1)
    else:
        qc.__getattribute__(name1)(qubit)
        qc.__getattribute__(name2)(qubit)
        qc.__getattribute__(name3)(qubit)

    return qc

# Sort gate identities by the number of qubits they act on`

def check_equivalence(circ1, circ2):
    """Checks if two circuits are equivalent using fidelity."""
    state1 = Statevector.from_instruction(circ1)
    state2 = Statevector.from_instruction(circ2)
    fidelity = state1.inner(state2).real ** 2
    return fidelity >= 0.99999, fidelity

def simplify_using_conjugation_only(gate_list):
    """Simplify using only conjugation rules."""
    simplified = []
    i = 0
    while i < len(gate_list):
        g1 = gate_list[i]

        if i + 2 < len(gate_list):
            g2 = gate_list[i + 1]
            g3 = gate_list[i + 2]
            result = apply_conjugation_rule(g1["name"], g2["name"], g3["name"])
            if result:
                target = g1.get("target") or (g1.get("qubits") or [None])[0]
                simplified.append({
                    "name": result.lstrip("-"),
                    "qubits": [target]
                })
                i += 3
                continue


        simplified.append(g1)
        i += 1

    return simplified

def test_all_conjugation_rules():  
    ### Tests all conjugation rules for correctness.
    for idx, rule in enumerate(CONJUGATION_RULES):
        print(f"\n[{idx+1}] Rule: {rule['gate1']} + {rule['gate2']} + {rule['gate3']} → {rule['replacement']}")
        try:
            original_circuit = create_circuit_for_conjugation(rule)
            print("Original Circuit:")
            print(original_circuit.draw())
            
            gate_list = convert_qiskit_to_gate_list(original_circuit)
            simplified_gate_list = simplify_using_conjugation_only(gate_list)
            simplified_circuit = convert_to_qiskit_circuit(simplified_gate_list, original_circuit.num_qubits)

            print("Simplified Circuit:")
            print(simplified_circuit.draw())

            
            equivalent, fidelity = check_equivalence(original_circuit, simplified_circuit)
            if equivalent:
                print(f" Circuits are equivalent (fidelity: {fidelity:.10f})")
            else:
                print(f" Circuits are NOT equivalent (fidelity: {fidelity:.10f})")  

        except Exception as e:
            print(f"Error testing rule {rule}: {e}")    

if __name__ == "__main__":
    test_all_conjugation_rules()              