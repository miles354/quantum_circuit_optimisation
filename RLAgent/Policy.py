# Add the parent directory to the Python path so Ican import local modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set matplotlib to non-interactive mode (until I sort GUI issues)
import matplotlib
matplotlib.use('Agg')

# Import required packages
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Import the custom quantum circuit environment
from RLEnvironment.RLGymEnvironment import QuantumCircuitEnv


# Wrapper to convert a MultiDiscrete action space into a flat Discrete space
class MultiDiscreteToDiscreteWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_action_space = env.action_space
        self.action_dims = self.original_action_space.nvec
        self.flat_action_space_size = int(np.prod(self.action_dims))
        self.action_space = gym.spaces.Discrete(self.flat_action_space_size)

    # Convert flat index back into multidimensional action
    def action(self, flat_action):
        return np.unravel_index(flat_action, self.action_dims)

    # Override step to accept flattened action
    def step(self, action):
        original_action = self.action(action)
        return self.env.step(original_action)


# Custom reset method to initialise the environment and return observation only
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)

    self.original_gate_list = self.generate_random_circuit()
    self.modified_gate_list = self.original_gate_list.copy()
    self.current_step = 0

    observation = self._get_observation()
    return observation  # Return only the observation (no info)


# Function to visualise a list of quantum gates as a circuit diagram
def visualize_gate_list(gate_list, title="Quantum Circuit", filename=None):
    num_qubits = 6  # Default number of qubits
    qc = QuantumCircuit(num_qubits)

    # Iterate through each gate and apply it to the circuit
    for gate_info in gate_list:
        if isinstance(gate_info, dict):
            gate = gate_info.get("gate", "")
            qubits = gate_info.get("qubits", [])
        else:
            gate = gate_info
            qubits = [0]

        if gate == 'null':
            continue  # Skip null gates

        try:
            # Apply different gates to the circuit based on name
            if gate == 'h':
                for q in qubits: qc.h(q)
            elif gate == 'x':
                for q in qubits: qc.x(q)
            elif gate == 'y':
                for q in qubits: qc.y(q)
            elif gate == 'z':
                for q in qubits: qc.z(q)
            elif gate == 'rx':
                for q in qubits: qc.rx(0.5, q)
            elif gate == 'ry':
                for q in qubits: qc.ry(0.5, q)
            elif gate == 'rz':
                for q in qubits: qc.rz(0.5, q)
            elif gate == 't':
                for q in qubits: qc.t(q)
            elif gate == 'tdg':
                for q in qubits: qc.tdg(q)
            elif gate == 's':
                for q in qubits: qc.s(q)
            elif gate == 'sdg':
                for q in qubits: qc.sdg(q)
            elif gate == 'u':
                for q in qubits: qc.u(0.5, 0.5, 0.5, q)
            elif gate == 'cx' and len(qubits) >= 2:
                qc.cx(qubits[0], qubits[1])
            elif gate == 'cz' and len(qubits) >= 2:
                qc.cz(qubits[0], qubits[1])
            elif gate == 'swap' and len(qubits) >= 2:
                qc.swap(qubits[0], qubits[1])
            elif gate == 'ccx' and len(qubits) >= 3:
                qc.ccx(qubits[0], qubits[1], qubits[2])
        except Exception as e:
            print(f"Skipping gate {gate} on qubits {qubits}: {e}")

    # Render and save the circuit diagram
    fig = circuit_drawer(qc, output='mpl')
    fig.suptitle(title)
    fig.tight_layout()

    if filename:
        fig.savefig(filename)
        print(f"Circuit saved as {filename}")
    else:
        default_filename = f"{title.replace(' ', '_').lower()}.png"
        fig.savefig(default_filename)
        print(f"Circuit saved as {default_filename}")

    plt.close(fig)  # Close plot to prevent memory leaks


# Factory function to initialise and wrap the environment
def make_env():
    env = QuantumCircuitEnv()
    env = MultiDiscreteToDiscreteWrapper(env)  # Wrap with action flattener
    env = DummyVecEnv([lambda: env])  # Make it compatible with stable-baselines3
    return env


# Main training and evaluation block
if __name__ == "__main__":
    os.makedirs("circuits", exist_ok=True)  # Create directory for saving circuit images

    env = make_env()
    check_env(env.envs[0], warn=True)  # Check environment compatibility with SB3

    # Initialise PPO model with MLP policy
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        verbose=1,
    )

    # Set up evaluation callback to save best models
    eval_env = make_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Train the model
    model.learn(total_timesteps=10_000, callback=eval_callback)
    model.save("ppo_quantum_rl")
    print("\nModel saved as 'ppo_quantum_rl.zip'")

    # Load trained model
    model = PPO.load("ppo_quantum_rl", env=env)

    episode_rewards = []

    # Run 5 test episodes using the trained agent
    for ep in range(5):
        obs = env.reset()   # Reset returns only observation from VecEnv
        obs = obs[0]        # Unwrap batch dimension
        info = {}           # No info provided by reset

        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step([action])  # VecEnv step
            obs = obs[0]
            reward = reward[0]
            done = done[0]
            info = info[0]

            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"\nEpisode {ep + 1} Total Reward: {total_reward}")

        original = info.get("original", [])
        modified = info.get("modified", [])

        print("Original Circuit:", original)
        print("Modified Circuit:", modified)

        # Save visualisations for original and modified circuits
        visualize_gate_list(original, title=f"Episode_{ep+1}_Original", filename=f"circuits/episode_{ep+1}_original.png")
        visualize_gate_list(modified, title=f"Episode_{ep+1}_Modified", filename=f"circuits/episode_{ep+1}_modified.png")

    # Plot and save total rewards for all episodes
    plt.figure()
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards")
    plt.savefig("episode_rewards.png")
    plt.close()
