import argparse
import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# Argtument parser setup
parser = argparse.ArgumentParser(description="Plot TensorBoard data with optional tag selection")
parser.add_argument('--tags', nargs='*', help='List of tags to plot (if not specified, all tags are plotted)')
parser.add_argument('--list-tags', action='store_true', help='List all available tags and exit')
parser.add_argument('--list-seeds', action='store_true', help='List all available seed folders and exit')
parser.add_argument('--seed', nargs='*', help='Seed number(s) (e.g., 197655404). Can specify multiple seeds to plot the same tags across experiments.')
parser.add_argument('--curve-labels', nargs='*', help='Custom labels for the curves (must match the number of seeds)')
args = parser.parse_args()

# Logdir path setup
base_log_path = "/Users/lucas/Desktop/DRP/MARL4DRP/epymarl/results/tb_logs/"

if args.list_seeds:
    print(" Available seed folders:")
    try:
        seeds = [d for d in os.listdir(base_log_path) if os.path.isdir(os.path.join(base_log_path, d))]
        for seed in sorted(seeds):
            print(f"  - {seed}")
    except FileNotFoundError:
        print("Logdir unavailable. Please check the path and try again.")
    exit(0)

if args.curve_labels and len(args.curve_labels) != len(args.seed):
    print("Error: The number of curve labels must match the number of seeds.")
    exit(1)

# Create seed to label mapping if custom labels provided
seed_to_label = {}
if args.curve_labels:
    seed_to_label = dict(zip(args.seed, args.curve_labels))


if not args.seed:
    print("Error: You must specify at least one seed number with --seed. Use --list-seeds to see available options.")
    exit(1)

if args.list_tags and len(args.seed) > 1:
    print("Error: --list-tags can only be used with a single seed.")
    exit(1)

# Collect data from all seeds
all_data = {}
for seed in args.seed:
    # Find the folder that contains the seed
    try:
        folders = [d for d in os.listdir(base_log_path) if os.path.isdir(os.path.join(base_log_path, d)) and f"_seed{seed}_" in d]
        if not folders:
            print(f"No folder found for seed {seed}.")
            continue
        elif len(folders) > 1:
            print(f"Multiple folders found for seed {seed}: {folders}")
            continue
        log_path = os.path.join(base_log_path, folders[0])
    except FileNotFoundError:
        print(f"Logdir unavailable for seed {seed}.")
        continue

    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()

    tags = ea.Tags()['scalars']

    if args.list_tags:
        print(f"Tags disponibles pour seed {seed} :")
        for tag in tags:
            print(f"  - {tag}")
        exit(0)

    # Filter tags
    if args.tags:
        selected_tags = [tag for tag in args.tags if tag in tags]
        if not selected_tags:
            print(f"None of the specified tags were found in the data for seed {seed}.")
            continue
    else:
        selected_tags = tags

    for tag in selected_tags:
        events = ea.Scalars(tag)
        label_base = seed_to_label.get(seed, f"seed {seed}")
        key = f"{tag} ({label_base})"
        all_data[key] = [(e.step, e.value) for e in events]

# Plotting
for label, values in all_data.items():
    steps = [v[0] for v in values]
    vals = [v[1] for v in values]
    plt.plot(steps, vals, label=label)

plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Value")
plt.title(f"{', '.join(args.tags)}")

# Save figure automatically
output_dir = "/Users/lucas/Desktop/DRP/MARL4DRP/results/plot_exports"
os.makedirs(output_dir, exist_ok=True)

# Build a safe output filename
seed_part = ",".join(seed_to_label.get(seed, seed).replace(" ", "-") for seed in args.seed)
tag_part = "-".join(args.tags) if args.tags else "all-tags"
output_path = os.path.join(output_dir, f"plot_{seed_part}_{tag_part}.png")

plt.savefig(output_path, bbox_inches='tight')
print(f"Graph saved to: {output_path}")

#plt.show()