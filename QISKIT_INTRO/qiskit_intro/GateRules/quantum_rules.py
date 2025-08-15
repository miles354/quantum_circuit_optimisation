import math

# Utility Functions

def angles_approximately_equal(a, b, tol=1e-6):
    # Compare two angles on the unit circle 
    return abs(((a - b + math.pi) % (2 * math.pi)) - math.pi) < tol

def gates_act_on_same_qubits(g1_info, g2_info):
    # Check if both gates have the same control and target qubits
    return (
        g1_info.get("control") == g2_info.get("control") and
        g1_info.get("target") == g2_info.get("target")
    )

def gates_act_on_same_all_qubits(g1_info, g2_info):
    # Check if both gates act on exactly the same set of qubits
    return set(g1_info.get("qubits", [])) == set(g2_info.get("qubits", []))

def acts_on_same_target(g1_info, g2_info):
    # Check if two gates share the same target qubit
    return g1_info.get("target") == g2_info.get("target")

def acts_on_same_qubit(g1_info, g2_info):
    # Check if both gates act on the same qubits (unordered)
    return set(g1_info.get("qubits", [])) == set(g2_info.get("qubits", []))

# Binary Gate Identities

# Rules to reduce or merge two consecutive gates
GATE_IDENTITIES = [
    {"gate1": "x", "gate2": "x", "replacement": ("null", "null")},
    {"gate1": "z", "gate2": "z", "replacement": ("null", "null")},
    {"gate1": "h", "gate2": "h", "replacement": ("null", "null")},
    {"gate1": "s", "gate2": "sdg", "replacement": ("null", "null")},
    {"gate1": "sdg", "gate2": "s", "replacement": ("null", "null")},
    {"gate1": "t", "gate2": "tdg", "replacement": ("null", "null")},
    {"gate1": "tdg", "gate2": "t", "replacement": ("null", "null")},
    {"gate1": "cx", "gate2": "cx", "replacement": ("null", "null"), "condition": "same_qubits"},
    {"gate1": "cz", "gate2": "cz", "replacement": ("null", "null"), "condition": "same_qubits"},
    {"gate1": "swap", "gate2": "swap", "replacement": ("null", "null"), "condition": "same_qubits"},
    {"gate1": "ccx", "gate2": "ccx", "replacement": ("null", "null"), "condition": "same_all_qubits"},
    {"gate1": "rz", "gate2": "rz", "replacement": ("merge", None)},
    {"gate1": "rx", "gate2": "rx", "replacement": ("merge", None)},
    {"gate1": "ry", "gate2": "ry", "replacement": ("merge", None)},
    {"gate1": "u", "gate2": "u", "replacement": ("check_inverse_u", None)},
]

def merge_rotation_angles(angle1, angle2):
    # Merge two angles and wrap into [-π, π]
    merged = (angle1 + angle2 + math.pi) % (2 * math.pi) - math.pi
    return None if abs(merged) < 1e-6 else merged

def apply_gate_identity(g1_info, g2_info):
    # Try applying identity rules to two gates
    for rule in GATE_IDENTITIES:
        if rule["gate1"] == g1_info["name"] and rule["gate2"] == g2_info["name"]:
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

# Conjugation Simplification Rules

# Transform gate sequences like H X H = Z
CONJUGATION_RULES = [
    {"gate1": "h", "gate2": "x", "gate3": "h", "replacement": "z"},
    {"gate1": "h", "gate2": "z", "gate3": "h", "replacement": "x"},
    {"gate1": "h", "gate2": "y", "gate3": "h", "replacement": "-y"},
    {"gate1": "s", "gate2": "x", "gate3": "sdg", "replacement": "y"},
    {"gate1": "sdg", "gate2": "x", "gate3": "s", "replacement": "-y"},
    {"gate1": "s", "gate2": "y", "gate3": "sdg", "replacement": "-x"},
    {"gate1": "sdg", "gate2": "y", "gate3": "s", "replacement": "x"},
]

def apply_conjugation_rule(pre, target, post):
    # Apply a known conjugation identity, if one matches
    for rule in CONJUGATION_RULES:
        if rule["gate1"] == pre and rule["gate2"] == target and rule["gate3"] == post:
            return rule["replacement"]
    return None

# Commutation and Anticommutation Rules

COMMUTE_RULES = [
    {"gate1": "x", "gate2": "rx", "condition": "same_target"},
    {"gate1": "rx", "gate2": "x", "condition": "same_target"},
    {"gate1": "z", "gate2": "rz", "condition": "same_target"},
    {"gate1": "rz", "gate2": "z", "condition": "same_target"},
    {"gate1": "rx", "gate2": "rz", "condition": "same_target"},
    {"gate1": "rz", "gate2": "rx", "condition": "same_target"},
    {"gate1": "ry", "gate2": "rz", "condition": "same_target"},
    {"gate1": "rz", "gate2": "ry", "condition": "same_target"},
    {"gate1": "s", "gate2": "z", "condition": "same_target"},
    {"gate1": "t", "gate2": "z", "condition": "same_target"},
    {"gate1": "s", "gate2": "t", "condition": "same_target"},
    {"gate1": "s", "gate2": "sdg", "condition": "same_target"},
    {"gate1": "t", "gate2": "tdg", "condition": "same_target"},
    {"gate1": "h", "gate2": "h", "condition": "same_target"},
    {"gate1": "cz", "gate2": "rz", "condition": "same_target"},
    {"gate1": "cx", "gate2": "rx", "condition": "same_target"},
    {"gate1": "cx", "gate2": "cz", "condition": "disjoint_qubits"},
    {"gate1": "cx", "gate2": "cx", "condition": "disjoint_qubits"},
    {"gate1": "cz", "gate2": "cz", "condition": "disjoint_qubits"},
    {"gate1": "swap", "gate2": "cx", "condition": "disjoint_qubits"},
    {"gate1": "swap", "gate2": "cz", "condition": "disjoint_qubits"},
    {"gate1": "cx", "gate2": "cx", "condition": "same_control_and_target"},
    {"gate1": "cz", "gate2": "cz", "condition": "same_control_and_target"},
]

ANTICOMMUTE_RULES = [
    {"gate1": "x", "gate2": "z", "condition": "same_qubit"},
    {"gate1": "z", "gate2": "x", "condition": "same_qubit"},
    {"gate1": "x", "gate2": "y", "condition": "same_qubit"},
    {"gate1": "y", "gate2": "x", "condition": "same_qubit"},
    {"gate1": "y", "gate2": "z", "condition": "same_qubit"},
    {"gate1": "z", "gate2": "y", "condition": "same_qubit"},
    {"gate1": "rx", "gate2": "rz", "condition": "same_qubit"},
    {"gate1": "rz", "gate2": "rx", "condition": "same_qubit"},
    {"gate1": "ry", "gate2": "rz", "condition": "same_qubit"},
    {"gate1": "rz", "gate2": "ry", "condition": "same_qubit"},
]

def check_commutation(g1_info, g2_info):

    if g1_info["name"] == g2_info["name"]:
        if gates_act_on_same_qubits(g1_info, g2_info):
            return "commute"
        if not set(g1_info.get("qubits", [])).intersection(set(g2_info.get("qubits", []))):
            return "commute"
    if acts_on_same_target(g1_info, g2_info):
        return "anticommute"

    return "none"

