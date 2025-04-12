import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------
# 1. Example "Tree" Definition and Random Rates
# ----------------------------------------------------
#
# We'll create a small, fixed "tree" by hand for clarity.
# Let's label internal nodes (N0, N1, N2) and tips/leaves (T0, T1, T2, T3).
#
#    (N2)
#    /  \
#  (N0) (N1)
#  / \   / \
# T0 T1 T2 T3
#
# We'll store it in an adjacency structure plus topological order.

nodes = ["N2", "N0", "T0", "T1", "N1", "T2", "T3"]
tree_edges = {
    "N2": ["N0", "N1"],
    "N0": ["T0", "T1"],
    "N1": ["T2", "T3"]
}

def get_children(node):
    return tree_edges.get(node, [])

def is_leaf(node):
    return node not in tree_edges

# We assign each node a random evolution rate (for demonstration).
np.random.seed(42)
node_rates = {node: np.random.uniform(0.5, 2.0) for node in nodes}

# ----------------------------------------------------
# 2. Compute Vertical Layout (Y positions)
# ----------------------------------------------------
def get_all_leaves():
    return [n for n in nodes if is_leaf(n)]

leaf_labels_sorted = sorted(get_all_leaves())

def compute_height(node):
    """
    Return the vertical position (Y) for each node.
    Leaves are placed at distinct Y positions; internal nodes at the average of their children.
    """
    if is_leaf(node):
        return float(leaf_labels_sorted.index(node))
    else:
        child_heights = [compute_height(c) for c in get_children(node)]
        return np.mean(child_heights)

node_y = {node: compute_height(node) for node in nodes}

# ----------------------------------------------------
# 3. Assign Baseline Ages (X positions) for demonstration
# ----------------------------------------------------
#
# We'll say the root (N2) is ~10, and each child is randomly 2..4 less, ignoring real logic.

base_ages = {}
base_ages["N2"] = 10.0

def assign_base_ages(node):
    children = get_children(node)
    if not children:
        # leaf: ~0 baseline
        base_ages[node] = 0.0
    else:
        node_age = base_ages[node]
        for c in children:
            # subtract 2..4
            base_ages[c] = node_age - np.random.uniform(2.0, 4.0)
            if base_ages[c] < 0:
                base_ages[c] = 0.0
            assign_base_ages(c)

assign_base_ages("N2")

# ----------------------------------------------------
# 4. Uncalibrated Age Function: nodes wiggle
# ----------------------------------------------------
def uncalibrated_age(node, t):
    """
    Return a fluctuating node age due to unknown calibrations.
    We'll use a sine function modulated by the node-specific rate.
    """
    base = base_ages[node]
    rate = node_rates[node]
    age = base + 1.0 * np.sin(rate * 0.1 * t)
    return max(age, 0.0)

# ----------------------------------------------------
# 5. Calibrated Age Function
# ----------------------------------------------------
calibration_points = {
    "N2": 10.0,  # Suppose the root is known at time=10
    "N1": 4.0,   # Suppose an internal node is anchored at time=4
}

def calibrated_age(node, t):
    """
    We define a linear interpolation from uncalibrated_age to a 'calibrated' age
    over some number of frames (total_frames). 
    """
    # If you're using frames=200 or 300 in the animation, match that below:
    total_frames = 100  
    fraction = min(t / total_frames, 1.0)

    uncal_age = uncalibrated_age(node, t)

    # final calibrated age (fallback to base_ages if not in calibration_points)
    if node in calibration_points:
        cal_age = calibration_points[node]
    else:
        cal_age = base_ages[node]

    # Linear blend from uncalibrated to calibrated
    age = (1.0 - fraction) * uncal_age + fraction * cal_age
    return max(age, 0.0)

# ----------------------------------------------------
# 6. Setup Matplotlib Figure/Artists
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
plt.title("Phylogenetic Tree – Uncalibrated & Calibrated Illustrations")

# We'll draw edges as line objects, and node points as scatter (via plot).
lines_dict = {}
for parent in tree_edges:
    for child in tree_edges[parent]:
        line_artist, = ax.plot([], [], lw=2, color='black')
        lines_dict[(parent, child)] = line_artist

node_points, = ax.plot([], [], 'o', markersize=6)
ax.set_xlim(0, 12)
ax.invert_xaxis()
ax.set_ylim(-1, len(leaf_labels_sorted))
ax.set_xlabel("Time (arbitrary units)")
ax.set_ylabel("Taxa / Node vertical offset")

# ----------------------------------------------------
# 7. Time Counter Text
# ----------------------------------------------------
time_text = ax.text(
    0.95, 0.95,
    '',
    transform=ax.transAxes,
    ha='right',
    va='top',
    fontsize=10,
    color='blue'
)

def init():
    """
    Initialize animation: clear data for lines and points,
    reset time_text to empty.
    """
    node_points.set_data([], [])
    for key in lines_dict:
        lines_dict[key].set_data([], [])
    time_text.set_text('')
    return [node_points] + list(lines_dict.values()) + [time_text]

# ----------------------------------------------------
# 8. Uncalibrated Animation
# ----------------------------------------------------
def animate_uncalibrated(frame):
    # Update node positions
    x_positions = []
    y_positions = []
    for node in nodes:
        x_positions.append(uncalibrated_age(node, frame))
        y_positions.append(node_y[node])
    node_points.set_data(x_positions, y_positions)

    # Update each edge
    for (parent, child), line_obj in lines_dict.items():
        px = uncalibrated_age(parent, frame)
        cx = uncalibrated_age(child, frame)
        py = node_y[parent]
        cy = node_y[child]
        line_obj.set_data([px, cx], [py, cy])

    # Update time counter
    # if interval=100ms, then each frame is 0.1s
    time_in_s = frame * 0.1  
    time_text.set_text(f"Uncalibrated Simulation Running: {time_in_s:.1f}s")

    return [node_points] + list(lines_dict.values()) + [time_text]

ani_uncal = FuncAnimation(
    fig,
    animate_uncalibrated,
    frames=200,   # Increase if you want a longer uncalibrated run
    interval=100, # ms
    init_func=init,
    blit=True,
    repeat=True
)

# ----------------------------------------------------
# 9. Calibrated Animation
# ----------------------------------------------------
def animate_calibrated(frame):
    x_positions = []
    y_positions = []
    for node in nodes:
        x_positions.append(calibrated_age(node, frame))
        y_positions.append(node_y[node])
    node_points.set_data(x_positions, y_positions)

    # Update edges
    for (parent, child), line_obj in lines_dict.items():
        px = calibrated_age(parent, frame)
        cx = calibrated_age(child, frame)
        py = node_y[parent]
        cy = node_y[child]
        line_obj.set_data([px, cx], [py, cy])

    # Update time counter
    time_in_s = frame * 0.1
    time_text.set_text(f"Calibrated Simulation Running: {time_in_s:.1f}s")

    return [node_points] + list(lines_dict.values()) + [time_text]

ani_cal = FuncAnimation(
    fig,
    animate_calibrated,
    frames=200,   # Increase if you want a longer calibrated run
    interval=100,
    init_func=init,
    blit=True,
    repeat=True
)

# ----------------------------------------------------
# 10. Show or Save the Animations
# ----------------------------------------------------
# Uncomment if you want an interactive display:
# plt.show()

# Example: Save each animation as an MP4 (requires ffmpeg installed).
# Each call will "re-run" the animation from frame 0 to frames=100 and save.
ani_uncal.save("uncalibrated_animation.mp4", writer="ffmpeg", fps=10)
ani_cal.save("calibrated_animation.mp4", writer="ffmpeg", fps=10)
ani_uncal.save("uncalibrated.gif", writer="imagemagick", fps=10)
ani_cal.save("calibrated.gif", writer="imagemagick", fps=10)

