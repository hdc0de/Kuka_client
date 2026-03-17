"""Compare real robot states (ep005) vs training data states (episode_000006)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load real robot states
real_states = np.load("/home/rl/projects/X-Sim/real_robot/debug_output/ep005/states.npy")  # (300, 8)

# Load training data states
df = pd.read_parquet(
    "/home/rl/projects/X-Sim/lerobot_datasets/training_data/greycup_on_bluecup_lite"
    "/data/chunk-000/episode_000006.parquet"
)
train_states = np.stack(df["observation.state"].values)  # (671, 8)

joint_names = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]

# Normalize time axis to [0, 1] for comparison
t_real  = np.linspace(0, 1, len(real_states))
t_train = np.linspace(0, 1, len(train_states))

fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
axes = axes.flatten()

for i, (ax, name) in enumerate(zip(axes, joint_names)):
    ax.plot(t_train, train_states[:, i], label="train ep006", color="steelblue", linewidth=1.5)
    ax.plot(t_real,  real_states[:, i],  label="real ep005",  color="tomato",    linewidth=1.5, linestyle="--")
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("State Comparison: Real ep005 vs Train ep006\n(x-axis normalized to episode length)", fontsize=13)
plt.tight_layout()
out = "/home/rl/projects/X-Sim/real_robot/debug_output/ep005/state_compare.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
