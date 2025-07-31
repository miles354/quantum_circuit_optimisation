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

from DatasetGeneration.circuit_utils import convert_to_qiskit_circuit, convert_qiskit_to_gate_list
from GateRules.quantum_rules import apply_gate_identity, apply_conjugation_rule, check_commutation
from DatasetGeneration.dataset_loader import load_circuit_from_dataset
from GateRules.quantum_rules import (
    angles_approximately_equal, 
    gates_act_on_same_qubits,
    gates_act_on_same_all_qubits,
    acts_on_same_target,
    acts_on_same_qubit,
    merge_rotation_angles, GATE_IDENTITIES
)

def simplify_gate_list(gate_list, max_iterations=1000):
    simplified = []
    i = 0
    iterations = 0
    while i < len(gate_list):
        if iterations > max_iterations:
            print("Max iterations reached in simplify_gate_list.")
            break

        g1 = gate_list[i]

        # Apply gate identity rules
        if i + 1 < len(gate_list):
            g2 = gate_list[i + 1]
            result = apply_gate_identity(g1, g2)
            if result:
                r1, _ = result
                if r1 != "null":
                    simplified.append(r1)  # Append the simplified gate
                i += 2  # Skip the next gate as it's cancelled or merged
                iterations += 1
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
                        iterations += 1
                        continue

        # Apply commutation and anticommutation simplifications
        if i + 1 < len(gate_list):
            g2 = gate_list[i + 1]
            relation = check_commutation(g1, g2)
            if relation == "commute":
                # Swap the gates to reduce depth
                gate_list[i], gate_list[i + 1] = g2, g1
                continue

        simplified.append(g1)
        i += 1
        iterations += 1
    return simplified




def apply_gate_identity(g1_info, g2_info):
    # Try applying identity rules to two gates
    for rule in GATE_IDENTITIES:
        if rule["gate1"] == g1_info["name"] and rule["gate2"] == g2_info["name"]:
            print(f"Applying identity rule: {rule['gate1']} + {rule['gate2']}")
            condition = rule.get("condition")
            # Apply condition checks if required
            if condition == "same_qubits" and not gates_act_on_same_qubits(g1_info, g2_info):
                continue
            if condition == "same_all_qubits" and not gates_act_on_same_all_qubits(g1_info, g2_info):
                continue

            action = rule["replacement"][0]

            if action == "merge":
                # Merge two rotations into one
                angle1 = g1_info.get("angle") or (g1_info.get("params", [0])[0])
                angle2 = g2_info.get("angle") or (g2_info.get("params", [0])[0])

                merged_angle = merge_rotation_angles(angle1, angle2)
                if merged_angle is None:
                    return ("null", "null")
                merged_gate = {
                    "name": g1_info["name"],
                    "angle": merged_angle,
                    "qubits": g1_info.get("qubits", []),
                    "control": g1_info.get("control"),
                    "target": g1_info.get("target")
                }
                return (merged_gate, "null")

            if action == "check_inverse_u":
                # Check if two U gates are inverses
                θ1, φ1, λ1 = g1_info.get("theta", 0), g1_info.get("phi", 0), g1_info.get("lam", 0)
                θ2, φ2, λ2 = g2_info.get("theta", 0), g2_info.get("phi", 0), g2_info.get("lam", 0)

                if (angles_approximately_equal(θ1, -θ2) and
                    angles_approximately_equal(φ1, -λ2) and
                    angles_approximately_equal(λ1, -φ2)):
                    return ("null", "null")

                if angles_approximately_equal(θ1, 0) and angles_approximately_equal(λ1, -φ1):
                    return ("null", "null")
                if angles_approximately_equal(θ2, 0) and angles_approximately_equal(λ2, -φ2):
                    return ("null", "null")

            return rule["replacement"]
    return None


def mutate_gate_list(gate_list, num_qubits):
    allowed_angles = [math.pi * n / 4 for n in range(1, 9)]  # List of allowed angles: π/4, π/2, 3π/4, π, ...
    new_gate_list = copy.deepcopy(gate_list)

    if not new_gate_list:
        return new_gate_list

    mutation_type = random.choice(["swap", "duplicate_cancel", "merge_gates", "commutative_swap", "perturb"])

    if mutation_type == "perturb":
        # Apply perturbations to the angle of rotation gates (Rx, Ry, Rz)
        idx = random.randint(0, len(new_gate_list) - 1)
        gate = new_gate_list[idx]
        
        if gate["name"] in ["rx", "ry", "rz"]:
            if "params" in gate and len(gate["params"]) > 0:  # Ensure parameters exist
                # Perturb the angle by selecting a random value from allowed_angles
                angle = gate["params"][0]  # Get the current angle
                perturbation = random.choice(allowed_angles)  # Choose a random allowed angle
                new_angle = (angle + perturbation) % (2 * math.pi)  # Ensure the angle is within valid range
                gate["params"] = [new_angle]  # Update the angle (list with one element)
                new_gate_list[idx] = gate
            else:
                # Skip gates that don't have parameters (like X, H, etc.)
                pass

    elif mutation_type == "swap":
        for i in range(len(new_gate_list) - 1):
            g1, g2 = new_gate_list[i], new_gate_list[i+1]
            q1 = set(g1.get("qubits", []) + [g1.get("target"), g1.get("control")]) - {None}
            q2 = set(g2.get("qubits", []) + [g2.get("target"), g2.get("control")]) - {None}
            if q1.isdisjoint(q2):
                new_gate_list[i], new_gate_list[i+1] = g2, g1
                break

    elif mutation_type == "duplicate_cancel":
        idx = random.randint(0, len(new_gate_list))
        qubit = random.randint(0, num_qubits - 1)  # Ensure valid qubit index
        new_gate_list.insert(idx, {"name": "x", "qubits": [qubit]})
        new_gate_list.insert(idx + 1, {"name": "x", "qubits": [qubit]})

    elif mutation_type == "merge_gates":
        # Merge gates if they are identical and compatible (e.g., consecutive RX gates)
        if len(new_gate_list) >= 2:
            g1, g2 = new_gate_list[0], new_gate_list[1]
            # Ensure both gates have parameters and are the same type
            if g1["name"] == g2["name"] and g1.get("params") and g2.get("params"):
                # Merge the angles (combine rotation angles)
                merged_angle = g1["params"][0] + g2["params"][0]  # Combine angles
                merged_gate = {
                    "name": g1["name"],
                    "params": [merged_angle],  # Update parameters
                    "qubits": g1.get("qubits", []),
                    "control": g1.get("control"),
                    "target": g1.get("target")
                }
                new_gate_list = [merged_gate] + new_gate_list[2:]

    elif mutation_type == "commutative_swap":
        # Swap commuting gates that do not affect the final result
        for i in range(len(new_gate_list) - 1):
            g1, g2 = new_gate_list[i], new_gate_list[i+1]
            relation = check_commutation(g1, g2)
            if relation == "commute":
                # Swap the gates to reduce depth
                new_gate_list[i], new_gate_list[i+1] = g2, g1
                break
    return new_gate_list


def cost_function(gate_list):
    qc = convert_to_qiskit_circuit(gate_list)
    return qc.depth() + 0.5 * len(gate_list)


def simulated_annealing_with_equivalence_check(initial_gate_list, num_qubits, T=1000.0, cooling_rate=0.95, max_iter=500, fidelity_threshold=0.9999):
    current_list = initial_gate_list
    current_cost = cost_function(current_list)
    best_list = current_list
    best_cost = current_cost

    # Convert the original circuit to Qiskit circuit to check fidelity
    original_qc = convert_to_qiskit_circuit(current_list, num_qubits)

    for _ in range(max_iter):
        # Mutate the gate list
        new_list = mutate_gate_list(current_list, num_qubits)
        simplified_list = simplify_gate_list(new_list)
        new_cost = cost_function(simplified_list)

        # Convert the new gate list to a Qiskit circuit to compare fidelity
        optimized_qc = convert_to_qiskit_circuit(simplified_list, num_qubits)
        state1 = Statevector.from_instruction(original_qc)
        state2 = Statevector.from_instruction(optimized_qc)
        fidelity = state1.inner(state2).real ** 2

        # If the fidelity drops below the threshold, revert to the original
        if fidelity >= fidelity_threshold:
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
    optimized_list = simulated_annealing_with_equivalence_check(gate_list, num_qubits=qiskit_circuit.num_qubits)
    return convert_to_qiskit_circuit(optimized_list, num_qubits=qiskit_circuit.num_qubits)


def check_circuit_equivalence(circ1, circ2):
    try:
        state1 = Statevector.from_instruction(circ1)
        state2 = Statevector.from_instruction(circ2)
        fidelity = state1.inner(state2).real ** 2
        print(f"\nFidelity between original and optimized circuits: {fidelity:.10f}")
        if fidelity > 0.99999:
            print("Circuits are functionally equivalent.")
        else:
            print("Circuits are NOT equivalent.")
    except Exception as e:
        print(f"Error comparing circuits: {e}")


def plot_depth_and_gate_comparison(depth_orig, depth_opt, gates_orig, gates_opt):
    save_path = os.path.join(
        os.path.dirname(__file__),
        "depth_and_gate_comparison.png"
    )

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].bar(["Original", "Optimized"], [depth_orig, depth_opt], color=["tab:blue", "tab:green"])
    axs[0].set_title("Circuit Depth")
    axs[0].set_ylabel("Depth")

    axs[1].bar(["Original", "Optimized"], [gates_orig, gates_opt], color=["tab:blue", "tab:green"])
    axs[1].set_title("Gate Count")
    axs[1].set_ylabel("Number of Gates")

    for ax, values in zip(axs, [(depth_orig, depth_opt), (gates_orig, gates_opt)]):
        for i, v in enumerate(values):
            ax.text(i, v + 0.2, str(v), ha='center')

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.show()


def show_bloch_spheres(qc, title_prefix="", filename=None):
    state = Statevector.from_instruction(qc)
    num_qubits = qc.num_qubits
    reduced = [partial_trace(state, [i for i in range(num_qubits) if i != j]) for j in range(num_qubits)]

    def get_components(dm):
        x = 2 * dm.data[0, 1].real
        y = 2 * dm.data[0, 1].imag
        z = dm.data[0, 0].real - dm.data[1, 1].real
        return [x, y, z]

    cols = 3
    rows = (num_qubits + cols - 1) // cols
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    fig.suptitle(f"{title_prefix}Bloch Spheres", fontsize=16)
    axes = [fig.add_subplot(rows, cols, i + 1, projection='3d') for i in range(num_qubits)]

    for i, (dm, ax) in enumerate(zip(reduced, axes)):
        b = Bloch(fig=fig, axes=ax)
        b.add_vectors(get_components(dm))
        b.render()
        ax.set_title(f"Qubit {i}")

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename)

    plt.show()


# === MAIN EXECUTION ===
if __name__ == "__main__":
    original = load_circuit_from_dataset(index=np.random.randint(0, 100))
    optimized = optimize_circuit_with_simulated_annealing(original)

    original_gate_list = convert_qiskit_to_gate_list(original)
    optimized_gate_list = convert_qiskit_to_gate_list(optimized)

    print("Depth of Original Circuit:", original.depth())
    print("Depth of Optimized Circuit:", optimized.depth())
    print("Number of Gates in Original Circuit:", len(original_gate_list))
    print("Number of Gates in Optimized Circuit:", len(optimized_gate_list))

    check_circuit_equivalence(original, optimized)

    # Display Circuits in Terminal
    print("\nOriginal Circuit:")
    print(original.draw())
    print("\nOptimized Circuit:")
    print(optimized.draw())

    # Create output directory
    benchmark_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(benchmark_dir, exist_ok=True)

    # Plot circuit metrics
    plot_depth_and_gate_comparison(
        original.depth(), optimized.depth(),
        len(original_gate_list), len(optimized_gate_list)
    )

    # === Show and save Bloch spheres ===
    show_bloch_spheres(
        original,
        title_prefix="Original ",
        filename=os.path.join(benchmark_dir, "bloch_original.png")
    )

    show_bloch_spheres(
        optimized,
        title_prefix="Optimized ",
        filename=os.path.join(benchmark_dir, "bloch_optimized.png")
    )

    # Visualize and save circuit diagrams
    fig_orig = circuit_drawer(original, output="mpl")
    fig_orig.savefig(os.path.join(benchmark_dir, "circuit_original.png"))
    plt.close(fig_orig)

    fig_opt = circuit_drawer(optimized, output="mpl")
    fig_opt.savefig(os.path.join(benchmark_dir, "circuit_optimized.png"))
    plt.close(fig_opt)
