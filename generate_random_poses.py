import json
import random
import math

# --------------------------
# Configurable parameters
# --------------------------

NUM_SAMPLES = 10       # how many random rover poses to generate
X_MIN, X_MAX = 0.0, 9.0
Y_MIN, Y_MAX = 0.0, 14.2

RANDOM_SEED = 42       # set to None for non-deterministic output


# --------------------------
# Pose generation
# --------------------------

def generate_pose(i):
    """Return one pose entry in the exact structure of rover.json."""
    x = random.uniform(X_MIN, X_MAX)
    y = random.uniform(Y_MIN, Y_MAX)
    yaw = random.uniform(0.0, 2 * math.pi)   # radians, as in your sample file

    return {
        "id": f"position_{i:03d}",
        "location": [x, y, 0],
        "rotation": [0, 0, yaw]
    }


def main():
    # Reproducibility if desired
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    rovers = [generate_pose(i) for i in range(1, NUM_SAMPLES + 1)]

    output = {"rovers": rovers}

    with open("./configs/rover.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {NUM_SAMPLES} poses to ./configs/rover.json")

if __name__ == "__main__":
    main()
