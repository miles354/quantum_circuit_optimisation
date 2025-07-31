import random
import math
import copy
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization.bloch import Bloch
from qiskit.visualization import circuit_drawer

# Add parent directory for custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from GateRules.quantum_rules import (
    apply_gate_identity,
    apply_conjugation_rule,
    check_commutation,
    angles_approximately_equal,
    gates_act_on_same_qubits,
    gates_act_on_same_all_qubits,
    acts_on_same_target,
    acts_on_same_qubit,
    merge_rotation_angles,
    GATE_IDENTITIES
)
from DatasetGeneration.circuit_utils import convert_qiskit_to_gate_list, convert_to_qiskit_circuit

# Utility function to create a tester circuit for each rule
def create_tester_circuit(rule_name):
    circuit = QuantumCircuit(2)

    if rule_name == 'identity':
        # Two consecutive X gates should cancel each other out
        circuit.x(0)  # X gate on qubit 0
        circuit.x(0)  # X gate on qubit 0 again (should cancel out)

    elif rule_name == 'commutation':
        # Apply Rx and Rz on qubit 0, which should commute
        circuit.rx(math.pi/2, 0)  # Rx gate on qubit 0
        circuit.rz(math.pi/2, 0)  # Rz gate on qubit 0 (commutes with Rx)

    elif rule_name == 'conjugation':
        # Apply H X H, which should convert X to Z due to conjugation
        circuit.h(0)  # Apply H gate
        circuit.x(0)  # Apply X gate
        circuit.h(0)  # Apply H gate again (should conjugate X to Z)

    elif rule_name == 'merging_rotations':
        # Apply two consecutive Rx gates with the same angle (should merge)
        circuit.rx(math.pi/4, 0)  # Apply Rx with pi/4
        circuit.rx(math.pi/4, 0)  # Apply Rx again (should merge to a single Rx with pi/2)

    return circuit

# Utility function to check equivalence of the original and simplified circuits
def check_equivalence(circ1, circ2):
    state1 = Statevector.from_instruction(circ1)
    state2 = Statevector.from_instruction(circ2)
    fidelity = state1.inner(state2).real ** 2
    if fidelity > 0.99999:
        print(f"Circuits are functionally equivalent with fidelity: {fidelity:.10f}")
    else:
        print(f"Circuits are NOT equivalent. Fidelity: {fidelity:.10f}")

# Simplify gate list based on the gate rules and convert back to a Qiskit circuit
def simplify_gate_list(gate_list, num_qubits=2):
    simplified = []
    i = 0
    while i < len(gate_list):
        g1 = gate_list[i]

        # Apply gate identity rules from GATE_IDENTITIES
        if i + 1 < len(gate_list):
            g2 = gate_list[i + 1]
            result = apply_gate_identity(g1, g2)
            if result:
                r1, _ = result
                if r1 != "null":
                    simplified.append(r1)  # Append the simplified gate
                i += 2  # Skip the next gate as it's cancelled or merged
                continue

        # Apply conjugation simplifications (H X H = Z)
        if i + 2 < len(gate_list):
            g2 = gate_list[i + 1]
            g3 = gate_list[i + 2]
            if (
                g1["name"] in ["h", "s", "sdg"]
                and g2["name"] in ["x", "y", "z"]
                and g3["name"] in ["h", "s", "sdg"]
            ):
                target = g1.get("target") or (g1.get("qubits") or [None])[0]
                if (
                    target == (g2.get("target") or (g2.get("qubits") or [None])[0])
                    and target == (g3.get("target") or (g3.get("qubits") or [None])[0])
                ):
                    rep = apply_conjugation_rule(g1["name"], g2["name"], g3["name"])
                    if rep:
                        rep_gate = {"name": rep.lstrip("-"), "qubits": [target]}
                        simplified.append(rep_gate)
                        i += 3
                        continue

        simplified.append(g1)
        i += 1

    # Convert the simplified gate list back to a Qiskit QuantumCircuit
    simplified_circuit = QuantumCircuit(num_qubits)
    for gate in simplified:
        name = gate.get("name")
        qubits = gate.get("qubits", [])
        params = gate.get("params", [])
        
        if name == "x":
            simplified_circuit.x(qubits[0])
        elif name == "rx":
            if params:
                simplified_circuit.rx(params[0], qubits[0])
        elif name == "rz":
            if params:
                simplified_circuit.rz(params[0], qubits[0])
        elif name == "h":
            simplified_circuit.h(qubits[0])
        elif name == "s":
            simplified_circuit.s(qubits[0])
        elif name == "sdg":
            simplified_circuit.sdg(qubits[0])

    return simplified_circuit

# Main function to run tests for all gate rules
def run_tests():
    # Test identity rule
    print("Testing identity rule...")
    circuit = create_tester_circuit('identity')
    print("Original Circuit (Identity Test):")
    print(circuit.draw())
    optimized_circuit = simplify_gate_list(convert_qiskit_to_gate_list(circuit))
    print("Simplified Circuit (Identity Test):")
    print(optimized_circuit.draw())
    check_equivalence(circuit, optimized_circuit)

    # Test commutation rule
    print("Testing commutation rule...")
    circuit = create_tester_circuit('commutation')
    print("Original Circuit (Commutation Test):")
    print(circuit.draw())
    optimized_circuit = simplify_gate_list(convert_qiskit_to_gate_list(circuit))
    print("Simplified Circuit (Commutation Test):")
    print(optimized_circuit.draw())
    check_equivalence(circuit, optimized_circuit)

    # Test conjugation rule
    print("Testing conjugation rule...")
    circuit = create_tester_circuit('conjugation')
    print("Original Circuit (Conjugation Test):")
    print(circuit.draw())
    optimized_circuit = simplify_gate_list(convert_qiskit_to_gate_list(circuit))
    print("Simplified Circuit (Conjugation Test):")
    print(optimized_circuit.draw())
    check_equivalence(circuit, optimized_circuit)

    # Test merging rotations rule
    print("Testing merging rotations rule...")
    circuit = create_tester_circuit('merging_rotations')
    print("Original Circuit (Merging Rotations Test):")
    print(circuit.draw())
    optimized_circuit = simplify_gate_list(convert_qiskit_to_gate_list(circuit))
    print("Simplified Circuit (Merging Rotations Test):")
    print(optimized_circuit.draw())
    check_equivalence(circuit, optimized_circuit)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    run_tests()
