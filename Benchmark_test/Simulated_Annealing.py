import random
import math
import copy
from qiskit import QuantumCircuit
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from circuit_utils import compute_depth, compute_naive_depth, convert_to_qiskit_circuit, convert_qiskit_to_gate_list
from GateRules.quantum_rules import apply_gate_identity, apply_conjugation_rule, check_commutation
from dataset_loader import load_circuit_from_dataset

def simplify_gate_list(gate_list):
    simplified = []
    i = 0
    while i < len(gate_list):
        g1 = gate_list[i]

        if i + 1 < len(gate_list):
            g2 = gate_list[i + 1]
            result = apply_gate_identity(g1, g2)
            if result:
                r1, r2 = result
                if r1 != "null":
                    simplified.append(r1)
                i += 2
                continue

        if i + 2 < len(gate_list):
            g2 = gate_list[i + 1]
            g3 = gate_list[i + 2]
            if (
                g1["name"] in ["h", "s", "sdg"] and
                g2["name"] in ["x", "y", "z"] and
                g3["name"] in ["h", "s", "sdg"]
            ):
                target = g1.get("target") or (g1.get("qubits") or [None])[0]
                if (
                    target == (g2.get("target") or (g2.get("qubits") or [None])[0]) and
                    target == (g3.get("target") or (g3.get("qubits") or [None])[0])
                ):
                    rep = apply_conjugation_rule(g1["name"], g2["name"], g3["name"])
                    if rep:
                        rep_gate = {"name": rep.lstrip('-'), "qubits": [target]}
                        simplified.append(rep_gate)
                        i += 3
                        continue

        simplified.append(g1)
        i += 1

    return simplified

def mutate_gate_list(gate_list):
    allowed_angles = [math.pi * n / 4 for n in range(1, 9)]
    new_gate_list = copy.deepcopy(gate_list)

    if not new_gate_list:
        return new_gate_list

    idx = random.randint(0, len(new_gate_list) - 1)
    gate = new_gate_list[idx]

    if gate["name"] in ["rx", "ry", "rz"]:
        gate["params"] = [random.choice(allowed_angles)]
    elif gate["name"] == "u":
        gate["params"] = [random.choice(allowed_angles) for _ in range(3)]
    elif idx < len(new_gate_list) - 1:
        g1 = new_gate_list[idx]
        g2 = new_gate_list[idx + 1]
        if set(g1["qubits"]).isdisjoint(set(g2["qubits"])):
            new_gate_list[idx], new_gate_list[idx + 1] = g2, g1

    new_gate_list[idx] = gate
    return new_gate_list

def cost_function(gate_list):
    qc = convert_to_qiskit_circuit(gate_list)
    return qc.depth() + 0.5 * len(gate_list)

def simulated_annealing(initial_gate_list, num_qubits, T=1000.0, cooling_rate=0.95, max_iter=500):
    current_list = initial_gate_list
    current_cost = cost_function(current_list)
    best_list = current_list
    best_cost = current_cost

    for _ in range(max_iter):
        new_list = mutate_gate_list(current_list)
        simplified_list = simplify_gate_list(new_list)
        new_cost = cost_function(simplified_list)
        delta = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / T):
            current_list = simplified_list
            current_cost = new_cost
            if new_cost < best_cost:
                best_list = simplified_list
                best_cost = new_cost

        T *= cooling_rate

    return best_list

def optimize_circuit_with_simulated_annealing(qiskit_circuit):
    gate_list = convert_qiskit_to_gate_list(qiskit_circuit)
    optimized_list = simulated_annealing(gate_list, num_qubits=qiskit_circuit.num_qubits)
    return convert_to_qiskit_circuit(optimized_list, num_qubits=qiskit_circuit.num_qubits)

if __name__ == "__main__":
    original = load_circuit_from_dataset(index=80)
    optimized = optimize_circuit_with_simulated_annealing(original)

    print("Original Circuit:")
    print(original)

    print("\nOptimized Circuit:")
    print(optimized)

    original_gate_list = convert_qiskit_to_gate_list(original)
    optimized_gate_list = convert_qiskit_to_gate_list(optimized)

    print("\nOriginal Gate List:")
    print(original_gate_list)

    print("\nOptimized Gate List:")
    print(optimized_gate_list)

    print("Depth of Original Circuit:", original.depth())
    print("Depth of Optimized Circuit:", optimized.depth())

    print("Number of Gates in Original Circuit:", len(original_gate_list))
    print("Number of Gates in Optimized Circuit:", len(optimized_gate_list))

    