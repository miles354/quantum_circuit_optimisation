import os
import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Add parent directory for custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from GateRules.quantum_rules import COMMUTE_RULES, ANTICOMMUTE_RULES
from DatasetGeneration.circuit_utils import convert_qiskit_to_gate_list, convert_to_qiskit_circuit


def acts_on_same_target(g1, g2):
    return (g1.get("target") == g2.get("target") or
            (g1.get("qubits") and g2.get("qubits") and g1["qubits"] == g2["qubits"]))


def acts_on_same_qubit(g1, g2):
    q1 = g1.get("qubits", [])
    q2 = g2.get("qubits", [])
    return bool(set(q1) & set(q2))


def create_circuit_for_commutation(rule):
    qc = QuantumCircuit(6)
    name1 = rule["gate1"]
    name2 = rule["gate2"]
    condition = rule.get("condition", "")
    theta = np.pi / 3
    q0, q1, q2, q3 = 0, 1, 2, 3

    def apply_gate(name, a, b=None):
        if name in ["rx", "ry", "rz"]:
            getattr(qc, name)(theta, a)
        elif name in ["cx", "cz", "swap"]:
            if b is None:
                raise ValueError(f"{name} gate needs two qubits")
            getattr(qc, name)(a, b)
        else:
            getattr(qc, name)(a)

    if "same_all_qubits" in condition:
        apply_gate(name1, q0)
        apply_gate(name2, q0)
        apply_gate(name1, q1)
        apply_gate(name2, q1)
    elif "same_target" in condition:
        apply_gate(name1, q0, q1 if name1 in ["cx", "cz", "swap"] else None)
        apply_gate(name2, q0, q1 if name2 in ["cx", "cz", "swap"] else None)
    elif "disjoint_qubits" in condition:
        apply_gate(name1, q0, q1 if name1 in ["cx", "cz", "swap"] else None)
        apply_gate(name2, q2, q3 if name2 in ["cx", "cz", "swap"] else None)
    elif "same_control_and_target" in condition:
        apply_gate(name1, q0, q1)
        apply_gate(name2, q0, q1)
    elif "same_qubit" in condition:
        apply_gate(name1, q0)
        apply_gate(name2, q0)
    else:
        apply_gate(name1, q0, q1 if name1 in ["cx", "cz", "swap"] else None)
        apply_gate(name2, q0, q1 if name2 in ["cx", "cz", "swap"] else None)

    return qc


def check_commutation_extended(g1_info, g2_info):
    for rule in COMMUTE_RULES:
        if {g1_info["name"], g2_info["name"]} == {rule["gate1"], rule["gate2"]}:
            cond = rule["condition"]
            if cond == "same_target" and acts_on_same_target(g1_info, g2_info):
                return "commute"
            if cond == "disjoint_qubits":
                all_q1 = set(g1_info.get("qubits", [])) | {g1_info.get("control"), g1_info.get("target")}
                all_q2 = set(g2_info.get("qubits", [])) | {g2_info.get("control"), g2_info.get("target")}
                if all_q1.isdisjoint(all_q2):
                    return "commute"
            if cond == "same_control_and_target":
                return g1_info.get("control") == g2_info.get("control") and g1_info.get("target") == g2_info.get("target")

    for rule in ANTICOMMUTE_RULES:
        if {g1_info["name"], g2_info["name"]} == {rule["gate1"], rule["gate2"]}:
            if rule["condition"] == "same_qubit" and acts_on_same_qubit(g1_info, g2_info):
                return "anticommute"

    return None


def simplify_commute_and_anticommute(gate_list):
    """Swaps gates if they commute or anticommute. Tracks phase if needed."""
    simplified = gate_list.copy()
    i = 0
    global_phase = 1

    while i + 1 < len(simplified):
        g1 = simplified[i]
        g2 = simplified[i + 1]
        relation = check_commutation_extended(g1, g2)

        print(f"Checking: {g1['name']} <-> {g2['name']} → {relation}")

        if relation == "commute":
            simplified[i], simplified[i + 1] = g2, g1
            print(f"  → Swapped (commute): {g1['name']} <-> {g2['name']}")
            i += 2
        elif relation == "anticommute":
            simplified[i], simplified[i + 1] = g2, g1
            global_phase *= -1
            print(f"  → Swapped (anticommute, phase *= -1): {g1['name']} <-> {g2['name']}")
            i += 2
        else:
            i += 1

    return simplified, global_phase


def check_equivalence(circ1, circ2, phase=1):
    """Checks equivalence (up to global phase)."""
    state1 = Statevector.from_instruction(circ1)
    state2 = Statevector.from_instruction(circ2)
    fidelity = abs(state1.inner(state2)) ** 2

    if phase == -1:
        state2 *= -1  # Apply global phase
        fidelity = abs(state1.inner(state2)) ** 2

    return fidelity >= 0.99999, fidelity


def test_all_commute_and_anticommute_rules():
    all_rules = COMMUTE_RULES + ANTICOMMUTE_RULES
    for idx, rule in enumerate(all_rules):
        print(f"\n[{idx + 1}] Rule: {rule}")
        try:
            original_circuit = create_circuit_for_commutation(rule)
            print("Original Circuit:")
            print(original_circuit.draw())

            gate_list = convert_qiskit_to_gate_list(original_circuit)
            simplified_gate_list, global_phase = simplify_commute_and_anticommute(gate_list)
            simplified_circuit = convert_to_qiskit_circuit(simplified_gate_list, original_circuit.num_qubits)

            print("Simplified Circuit:")
            print(simplified_circuit.draw())

            equivalent, fidelity = check_equivalence(original_circuit, simplified_circuit, global_phase)
            status = "passed" if equivalent else "FAILED"
            print(f"→ Rule {idx + 1} {status} with fidelity: {fidelity:.6f} (phase: {'-1' if global_phase == -1 else '+1'})")

        except Exception as e:
            print(f"Error processing rule {idx + 1}: {e}")


if __name__ == "__main__":
    test_all_commute_and_anticommute_rules()
