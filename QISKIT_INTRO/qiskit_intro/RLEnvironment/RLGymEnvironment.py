import sys
import os
import gymnasium as gym
import numpy as np
import json
import math
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from GateRules.quantum_rules import (
    apply_gate_identity,
    apply_conjugation_rule,
    check_commutation,
)


class QuantumCircuitEnv(gym.Env):
    def __init__(self, dataset_path="quantum_circuit_dataset_with_native.json", max_gates=50, max_steps=100, num_qubits=6):
        super().__init__()

        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)

        self.max_gates = max_gates
        self.max_steps = max_steps
        self.num_qubits = num_qubits

        self.current_step = 0
        self.original_gate_list = []
        self.modified_gate_list = []

        self.original_native_depth = 0
        self.original_native_gate_count = 0

        self.action_types = {
            0: "delete",
            1: "replace",
            2: "keep",
            3: "swap",
            4: "cancel",
            5: "commute"
        }

        self.gate_map = {g: i for i, g in enumerate([
            'h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 't', 'tdg', 's', 'sdg', 'u',
            'cx', 'cz', 'swap', 'ccx', 'null'
        ])}
        self.reverse_gate_map = {i: g for g, i in self.gate_map.items()}

        self.action_space = gym.spaces.MultiDiscrete([
            len(self.action_types),
            self.max_gates,
            len(self.gate_map)
        ])

        self.observation_space = gym.spaces.Box(
            low=0,
            high=2000,
            shape=(self.max_gates + 4,),
            dtype=np.float32
        )

    def _get_gate_name(self, gate):
        return gate.get("name", "null")

    def _can_commute(self, g1_info, g2_info):
        # Ensure both gates have 'name' key
        if "name" not in g1_info or "name" not in g2_info:
            return False
        return check_commutation(g1_info, g2_info) == "commute"

    def apply_identity_or_conjugation(self, gate_info):
        result = apply_gate_identity(gate_info, gate_info)
        if result and result[0] != "null":
            return result[0]
        conj = apply_conjugation_rule(gate_info.get("name", ""), gate_info.get("name", ""), gate_info.get("name", ""))
        if conj:
            return {"name": conj, "qubits": gate_info.get("qubits", [])}
        return gate_info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.original_gate_list = self.generate_random_circuit()
        self.modified_gate_list = copy.deepcopy(self.original_gate_list)
        self.current_step = 0

        return self._get_observation(), {}

    def step(self, action):
        action_type, gate_idx, new_gate_idx = action
        reward = 0.0
        done = False

        if gate_idx >= len(self.modified_gate_list) or not self.modified_gate_list:
            reward -= 1.0
        else:
            gate = self.modified_gate_list[gate_idx]

            if action_type == 0:  # delete
                self.modified_gate_list.pop(gate_idx)
                reward += 1.0

            elif action_type == 1:  # replace
                new_gate = self.reverse_gate_map[new_gate_idx]
                if new_gate != "null":
                    gate["name"] = new_gate 
                    simplified = self.apply_identity_or_conjugation(gate)
                    self.modified_gate_list[gate_idx] = simplified
                    reward += 0.7
                else:
                    reward -= 0.5

            elif action_type == 2:  # keep
                simplified = self.apply_identity_or_conjugation(gate)
                self.modified_gate_list[gate_idx] = simplified
                reward += 0.3

            elif action_type == 3:  # swap
                if gate_idx + 1 < len(self.modified_gate_list):
                    self.modified_gate_list[gate_idx], self.modified_gate_list[gate_idx + 1] = (
                        self.modified_gate_list[gate_idx + 1],
                        self.modified_gate_list[gate_idx]
                    )
                    reward += 0.3
                else:
                    reward -= 0.2

            elif action_type == 4:  # cancel
                if gate_idx + 1 < len(self.modified_gate_list):
                    next_gate = self.modified_gate_list[gate_idx + 1]
                    if self._get_gate_name(gate) == self._get_gate_name(next_gate):
                        del self.modified_gate_list[gate_idx:gate_idx + 2]
                        reward += 2.0
                    else:
                        reward -= 0.5
                else:
                    reward -= 0.5

            elif action_type == 5:  # commute
                if gate_idx + 1 < len(self.modified_gate_list):
                    if self._can_commute(gate, self.modified_gate_list[gate_idx + 1]):
                        self.modified_gate_list[gate_idx], self.modified_gate_list[gate_idx + 1] = (
                            self.modified_gate_list[gate_idx + 1],
                            self.modified_gate_list[gate_idx]
                        )
                        reward += 0.5
                    else:
                        reward -= 0.5
                else:
                    reward -= 0.5

        self.current_step += 1
        done = self.current_step >= self.max_steps

        if done:
            original_len = len(self.original_gate_list)
            final_len = len(self.modified_gate_list)
            reward += (original_len - final_len) * 0.2

        return self._get_observation(), reward, done, False, {
            "original": self.original_gate_list,
            "modified": self.modified_gate_list
        }

    def generate_random_circuit(self):
        if not self.dataset:
            return []
        return copy.deepcopy(np.random.choice(self.dataset)["circuit_diagram"])

    def _get_observation(self):
        obs = np.zeros(self.max_gates + 4, dtype=np.float32)

        for i, gate in enumerate(self.modified_gate_list[:self.max_gates]):
            name = self._get_gate_name(gate)
            idx = self.gate_map.get(name, self.gate_map["null"])
            obs[i] = idx

        obs[self.max_gates] = len(self.modified_gate_list)
        obs[self.max_gates + 1] = self.current_step
        obs[self.max_gates + 2] = self.original_native_depth
        obs[self.max_gates + 3] = self.original_native_gate_count

        return obs
