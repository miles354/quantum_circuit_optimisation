import sys
import os

# Add the parent directory to the system path so custom modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import numpy as np
import json
import math
from qiskit import QuantumCircuit

# Import quantum gate simplification rules from custom module
from GateRules.quantum_rules import (
    apply_gate_identity,
    apply_conjugation_rule,
    check_commutation,
)


class QuantumCircuitEnv(gym.Env):
    def __init__(self, dataset_path="quantum_circuit_dataset_with_native.json", max_gates=50, max_steps=100, num_qubits=6):
        super().__init__()

        # Load circuit dataset from JSON
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)

        self.max_gates = max_gates  # Maximum gates in observation
        self.max_steps = max_steps  # Maximum steps allowed in an episode
        self.num_qubits = num_qubits  # Number of qubits per circuit

        self.current_step = 0  # Current step index
        self.current_index = 0  # Current index in dataset

        # Define action types (the agent chooses one of these per step)
        self.action_types = {
            0: "delete",
            1: "replace",
            2: "keep",
            3: "swap",
            4: "cancel",
            5: "commute"
        }
        self.num_action_types = len(self.action_types)

        # Map gate names to integer indices for compact representation
        self.gate_map = {g: i for i, g in enumerate([
            'h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 't', 'tdg', 's', 'sdg', 'u',
            'cx', 'cz', 'swap', 'ccx', 'null'
        ])}
        # Reverse mapping from integer index to gate name
        self.reverse_gate_map = {i: g for g, i in self.gate_map.items()}

        # Define the action space as a triple: (action_type, gate_idx, new_gate_type)
        self.action_space = gym.spaces.MultiDiscrete([
            self.num_action_types,
            self.max_gates,
            len(self.gate_map)
        ])

        # Define the observation space: vector of gate indices + 4 extra features
        self.observation_space = gym.spaces.Box(
            low=0,
            high=2000,
            shape=(self.max_gates + 4,),
            dtype=np.float32
        )

        # Store the original and modified circuits
        self.original_gate_list = []
        self.modified_gate_list = []

        # Store original circuit statistics
        self.original_native_depth = 0
        self.original_native_gate_count = 0

    def angles_approximately_equal(self, a, b, tol=1e-6):
        # Helper for checking if two angles are approximately equal (2pi)
        return abs(((a - b + math.pi) % (2 * math.pi)) - math.pi) < tol

    def _can_commute(self, g1_info, g2_info):
        # Check if two gates commute using imported rule
        return check_commutation(g1_info, g2_info) == "commute"

    def apply_identity_or_conjugation(self, gate_info):
        """
        Try to simplify or transform the gate using identity or conjugation rules.
        Return the new gate info if a transformation is applied; otherwise, None.
        """
        identity_applied = apply_gate_identity(gate_info)
        if identity_applied:
            return identity_applied
        
        conjugated_gate = apply_conjugation_rule(gate_info)
        return conjugated_gate
    
    def reset(self, *, seed=None, options=None):
        # Reset environment state at the beginning of an episode
        super().reset(seed=seed)

        # Sample a new original circuit from the dataset
        self.original_circuit = self.generate_random_circuit()

        # Start with a copy of the original gate list as the modifiable one
        self.modified_gate_list = self.original_gate_list.copy()
        self.current_step = 0

        observation = self._get_observation()
        info = {
            "original": self.original_circuit,
            "modified": self.modified_gate_list
        }

        return observation, info

    def step(self, action):
        # Execute one step in the environment using the provided action
        action_type, gate_idx, new_gate_idx = action
        done = False
        reward = 0.00

        # If the action targets an invalid gate index
        if gate_idx >= len(self.modified_gate_list) or len(self.modified_gate_list) == 0:
            reward -= 1.00
        else:
            gate = self.modified_gate_list[gate_idx]

            if action_type == 0:  # delete
                self.modified_gate_list.pop(gate_idx)
                reward += 1.00

            elif action_type == 1:  # replace
                new_gate_name = self.reverse_gate_map[new_gate_idx]
                if new_gate_name != "null":
                    gate["gate"] = new_gate_name
                    # Try to simplify gate after replacement
                    simplified_gate = self.apply_identity_or_conjugation(gate)
                    if simplified_gate:
                        self.modified_gate_list[gate_idx] = simplified_gate
                        reward += 0.2  # Bonus reward for simplification
                    reward += 0.5
                else:
                    reward -= 0.5

            elif action_type == 2:  # keep
                # Try to simplify current gate
                simplified_gate = self.apply_identity_or_conjugation(gate)
                if simplified_gate:
                    self.modified_gate_list[gate_idx] = simplified_gate
                    reward += 0.2  # Bonus reward for simplification
                reward += 0.1

            elif action_type == 3:  # swap with next gate
                if gate_idx + 1 < len(self.modified_gate_list):
                    self.modified_gate_list[gate_idx], self.modified_gate_list[gate_idx + 1] = (
                        self.modified_gate_list[gate_idx + 1],
                        self.modified_gate_list[gate_idx],
                    )
                    reward += 0.2
                else:
                    reward -= 0.2

            elif action_type == 4:  # cancel with next gate
                if gate_idx + 1 < len(self.modified_gate_list):
                    next_gate = self.modified_gate_list[gate_idx + 1]
                    if gate["gate"] == next_gate["gate"]:
                        self.modified_gate_list.pop(gate_idx)
                        self.modified_gate_list.pop(gate_idx)
                        reward += 2.0
                    else:
                        reward -= 0.5
                else:
                    reward -= 0.5

            elif action_type == 5:  # commute with next gate
                if gate_idx + 1 < len(self.modified_gate_list):
                    if self._can_commute(gate, self.modified_gate_list[gate_idx + 1]):
                        self.modified_gate_list[gate_idx], self.modified_gate_list[gate_idx + 1] = (
                            self.modified_gate_list[gate_idx + 1],
                            self.modified_gate_list[gate_idx],
                        )
                        reward += 0.3
                    else:
                        reward -= 0.5
                else:
                    reward -= 0.5

            else:
                reward -= 1.0  # Invalid action type

        self.current_step += 1

        # End episode if max steps reached
        if self.current_step >= self.max_steps:
            done = True

        observation = self._get_observation()

        info = {
            "original": self.original_circuit,
            "modified": self.modified_gate_list
        }

        return observation, reward, done, False, info

    def generate_random_circuit(self):
        # If the dataset is not empty
        if len(self.dataset) > 0:
            # Randomly select a circuit from the dataset
            idx = np.random.randint(0, len(self.dataset))
            # Returns "circuit_diagram" from the selected dataset entry
            return self.dataset[idx]["circuit_diagram"]
        else:
            # If dataset is empty, return an empty circuit
            return []

    def _get_observation(self):
        # Initialise observation vector with zeros
        # Observation Space includes max_gates gate and 4 additional fields 
        obs = np.zeros(self.max_gates + 4, dtype=np.float32)

        # Fill the observation with gate indices from modified gate list
        for i, gate in enumerate(self.modified_gate_list):
            if i >= self.max_gates:
                break # Stop if reached max gates limit
            # Get gate name and convert to index
            gate_name = gate.get("gate", gate.get("name", "null"))
            #Map the gate name to corresponding index
            gate_idx = self.gate_map.get(gate_name, self.gate_map["null"])
            obs[i] = gate_idx # Store gate index in observation

        # Fill additional fields in the observation
        obs[self.max_gates] = len(self.modified_gate_list)
        obs[self.max_gates + 1] = self.current_step
        obs[self.max_gates + 2] = self.original_native_depth
        obs[self.max_gates + 3] = self.original_native_gate_count

        return obs # Return full observation vector
