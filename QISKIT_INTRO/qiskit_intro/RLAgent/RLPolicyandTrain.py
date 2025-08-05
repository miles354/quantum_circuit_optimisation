import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from RLEnvironment.RLGymEnvironment import QuantumCircuitEnv


class MultiDiscreteToDiscreteWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_action_space = env.action_space
        self.action_dims = self.original_action_space.nvec
        self.flat_action_space_size = int(np.prod(self.action_dims))
        self.action_space = gym.spaces.Discrete(self.flat_action_space_size)

    def action(self, flat_action):
        return np.unravel_index(flat_action, self.action_dims)

    def step(self, action):
        original_action = self.action(action)
        return self.env.step(original_action)


def visualize_gate_list(gate_list, title="Quantum Circuit", filename=None):
    num_qubits = 6
    qc = QuantumCircuit(num_qubits)

    for gate_info in gate_list:
        gate = gate_info.get("name", "null")
        qubits = gate_info.get("qubits", [])

        if gate == 'null':
            continue

        try:
            if gate == 'h': [qc.h(q) for q in qubits]
            elif gate == 'x': [qc.x(q) for q in qubits]
            elif gate == 'y': [qc.y(q) for q in qubits]
            elif gate == 'z': [qc.z(q) for q in qubits]
            elif gate == 'rx': [qc.rx(0.5, q) for q in qubits]
            elif gate == 'ry': [qc.ry(0.5, q) for q in qubits]
            elif gate == 'rz': [qc.rz(0.5, q) for q in qubits]
            elif gate == 't': [qc.t(q) for q in qubits]
            elif gate == 'tdg': [qc.tdg(q) for q in qubits]
            elif gate == 's': [qc.s(q) for q in qubits]
            elif gate == 'sdg': [qc.sdg(q) for q in qubits]
            elif gate == 'u': [qc.u(0.5, 0.5, 0.5, q) for q in qubits]
            elif gate == 'cx' and len(qubits) >= 2: qc.cx(qubits[0], qubits[1])
            elif gate == 'cz' and len(qubits) >= 2: qc.cz(qubits[0], qubits[1])
            elif gate == 'swap' and len(qubits) >= 2: qc.swap(qubits[0], qubits[1])
            elif gate == 'ccx' and len(qubits) >= 3: qc.ccx(qubits[0], qubits[1], qubits[2])
        except Exception as e:
            print(f"Skipping gate {gate} on qubits {qubits}: {e}")

    fig = circuit_drawer(qc, output='mpl')
    fig.suptitle(title)
    fig.tight_layout()
    if filename:
        fig.savefig(filename)
        print(f"Saved: {filename}")
    plt.close(fig)


def make_env(check=False):
    dataset_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "DatasetGeneration", "converted_rlgym_dataset.json"
    ))

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    base_env = QuantumCircuitEnv(dataset_path=dataset_path)
    wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

    if check:
        check_env(wrapped_env, warn=True)

    return DummyVecEnv([lambda: wrapped_env])


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    outputs_dir = os.path.join(base_dir, "outputs")
    circuits_dir = os.path.join(outputs_dir, "circuits")
    logs_dir = os.path.join(outputs_dir, "logs")
    checkpoints_dir = os.path.join(outputs_dir, "checkpoints")

    os.makedirs(circuits_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    env = make_env(check=True)

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

    eval_env = make_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoints_dir,
        log_path=logs_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=10000, callback=eval_callback)
    model.save(os.path.join(outputs_dir, "ppo_quantum_rl"))

    print("\nModel saved.")

    # Evaluate
    model = PPO.load(os.path.join(outputs_dir, "ppo_quantum_rl"), env=env)

    episode_rewards = []
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step([action])
            obs = obs[0]
            reward = reward[0]
            done = done[0]
            info = info[0]
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"\nEpisode {ep+1} - Reward: {total_reward}")
        visualize_gate_list(info["original"], title=f"Original_{ep+1}", filename=os.path.join(circuits_dir, f"original_{ep+1}.png"))
        visualize_gate_list(info["modified"], title=f"Modified_{ep+1}", filename=os.path.join(circuits_dir, f"modified_{ep+1}.png"))

    # Plot
    plt.figure()
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards")
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "episode_rewards.png"))
    plt.close()
