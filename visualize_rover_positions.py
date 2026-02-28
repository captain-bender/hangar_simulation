import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# Parse command line arguments
use_splits = '--splits' in sys.argv or '-s' in sys.argv
exclude_not_included = '--exclude-not-included' in sys.argv or '-e' in sys.argv

# Load the rover configuration
with open('configs/rover.json', 'r') as f:
    data = json.load(f)

# Extract positions and rotations
rovers = data['rovers']
x_positions = []
y_positions = []
yaw_angles = []
rover_ids = []

for rover in rovers:
    location = rover['location']
    rotation = rover['rotation']
    
    x_positions.append(location[0])
    y_positions.append(location[1])
    # Yaw is the third element of rotation (z-axis rotation)
    yaw_angles.append(rotation[2])
    rover_ids.append(rover['id'])

x_positions = np.array(x_positions)
y_positions = np.array(y_positions)
yaw_angles = np.array(yaw_angles)

# Load CSV data if splits feature is enabled
split_labels = None
if use_splits:
    try:
        df = pd.read_csv('dataset_version5_positions.csv')
        # Extract position numbers from image names and map to splits
        split_map = {}
        for _, row in df.iterrows():
            image_name = row['image_name']
            # Extract position number from image_name (e.g., position_002 from position_002_png.rf...)
            position_num = image_name.split('_')[1]
            split_map[position_num] = row['split']
        
        # Map splits to rover positions
        split_labels = []
        for rover_id in rover_ids:
            # Extract position number from rover_id (e.g., position_001)
            pos_num = rover_id.split('_')[1]
            split = split_map.get(pos_num, 'Not included')
            split_labels.append(split)
        
        split_labels = np.array(split_labels)
        print(f"Loaded split information from CSV")
        print(f"Train: {np.sum(split_labels == 'train')}, Validation: {np.sum((split_labels == 'valid') | (split_labels == 'val') | (split_labels == 'validation'))}, Test: {np.sum(split_labels == 'test')}, Not included: {np.sum(split_labels == 'Not included')}")
        
        # Exclude "Not included" entries if requested
        if exclude_not_included:
            mask = split_labels != 'Not included'
            x_positions = x_positions[mask]
            y_positions = y_positions[mask]
            yaw_angles = yaw_angles[mask]
            rover_ids = np.array(rover_ids)[mask]
            split_labels = split_labels[mask]
            print(f"Excluding 'Not included' positions. Reduced to {len(x_positions)} positions")
    except Exception as e:
        print(f"Warning: Could not load CSV data: {e}")
        use_splits = False

# Calculate direction vectors for arrows
arrow_length = 0.3
u = arrow_length * np.cos(yaw_angles)
v = arrow_length * np.sin(yaw_angles)

# Create the figure
fig, ax = plt.subplots(figsize=(14, 10))

# Plot the rover centers as points
if use_splits and split_labels is not None:
    # Color by split type
    color_map = {'train': 'blue', 'val': 'orange', 'valid': 'orange', 'validation': 'orange', 'test': 'green', 'Not included': 'gray'}
    unique_splits = np.unique(split_labels)
    
    for split in unique_splits:
        mask = split_labels == split
        color = color_map.get(split, 'gray')
        label = 'Validation' if split in ['val', 'valid', 'validation'] else split.capitalize()
        ax.scatter(x_positions[mask], y_positions[mask], c=color, s=20, alpha=0.6, label=label)
else:
    ax.scatter(x_positions, y_positions, c='blue', s=20, alpha=0.6, label='Rover Centers')

# Plot direction arrows using quiver
ax.quiver(x_positions, y_positions, u, v, 
          angles='xy', scale_units='xy', scale=1, 
          color='red', width=0.003, alpha=0.7, label='Direction')

# Set equal aspect ratio and labels
ax.set_aspect('equal')
ax.set_xlabel('X Position (meters)')
ax.set_ylabel('Y Position (meters)')
ax.set_title(f'Rover Positions and Directions ({len(x_positions)}/{len(rovers)} points)')
ax.grid(True, alpha=0.3)
ax.legend()

# Add text showing count
# ax.text(0.02, 0.98, f'Total Rovers: {len(rovers)}', 
#         transform=ax.transAxes, verticalalignment='top',
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('rover_positions_visualization.png', dpi=150, bbox_inches='tight')
print(f"Visualized {len(rovers)} rover positions")
if use_splits:
    print("Visualization includes dataset split coloring (Train=Blue, Validation=Orange, Test=Green, Not included=Gray)")
else:
    print("Tip: Run with --splits or -s flag to color by dataset split: python visualize_rover_positions.py --splits")
    print("Or use --exclude-not-included or -e to hide positions not in the dataset (requires --splits)")
print("Graph saved as 'rover_positions_visualization.png'")
plt.show()
