import re
import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_reward(log_file):
    # Read and parse the log file
    rewards = []
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'Reward\s+mean=([\d.]+)\s+std=([\d.]+)\s+min=([\d.]+)\s+max=([\d.]+)', line)
            if match:
                rewards.append({
                    'mean': float(match.group(1)),
                    'std': float(match.group(2)),
                    'min': float(match.group(3)),
                    'max': float(match.group(4))
                })

    # Create plot
    fig, ax = plt.subplots(figsize=(15, 6))

    x = range(len(rewards))
    means = [r['mean'] for r in rewards]
    stds = [r['std'] for r in rewards]
    mins = [r['min'] for r in rewards]
    maxs = [r['max'] for r in rewards]

    # Plot mean line
    ax.plot(x, means, 'b-', linewidth=1, label='Mean')

    # Plot shaded ±2 std area
    upper = [means[i] + 2*stds[i] for i in range(len(means))]
    lower = [means[i] - 2*stds[i] for i in range(len(means))]
    ax.fill_between(x, lower, upper, alpha=0.3, label='±2 std')

    # Plot min and max lines
    # ax.plot(x, mins, 'r--', linewidth=1, label='Min')
    # ax.plot(x, maxs, 'g--', linewidth=1, label='Max')

    # Labels and styling
    ax.set_xlabel('Batch number')
    ax.set_ylabel('Reward')
    ax.set_title('Reward statistics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reward_plot.png', dpi=150)
    plt.show()

    print(f"Processed {len(rewards)} reward entries")

if __name__ == "__main__":
    # log_file = "outputs/es_em_bad_medical_advice/em_nccl_20260321_184950/train.log"
    log_file = "outputs/es_em_bad_medical_advice/em_nccl_20260323_112605/train.log"
    plot_reward(log_file)