import json
import math

def blender_to_yolo_position(x_b, y_b):
    """
    Convert (x_B, y_B) from Blender to YOLO/camera frame.
    Simple swap, no sign flip:
        x_Y = y_B
        y_Y = x_B
    """
    x_y = y_b
    y_y = x_b
    return x_y, y_y

def blender_to_yolo_yaw(theta_b):
    """
    Convert yaw from Blender to YOLO/camera frame,
    with result in [0, 2π).

    YOLO yaw = Blender yaw - 90° = theta_b - π/2, wrapped.
    """
    theta_y = theta_b - math.pi / 2.0       # shift by -90°
    theta_y = theta_y % (2 * math.pi)       # wrap to [0, 2π)
    return theta_y


def convert_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rovers_in = data.get("rovers", [])
    rovers_out = []

    for rover in rovers_in:
        loc = rover.get("location", [0, 0, 0])
        rot = rover.get("rotation", [0, 0, 0])

        x_b, y_b, z_b = loc
        _, _, theta_b = rot

        # Convert pose to YOLO frame
        x_y, y_y = blender_to_yolo_position(x_b, y_b)
        theta_y = blender_to_yolo_yaw(theta_b)

        rover_out = {
            "id": rover.get("id", ""),
            "location": [x_y, y_y, z_b],
            "rotation": [0, 0, theta_y]
        }

        rovers_out.append(rover_out)

    new_data = {"rovers": rovers_out}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)

    print(f"Converted {len(rovers_out)} rover poses (Blender -> YOLO frame).")


if __name__ == "__main__":
    convert_file("./configs/rover.json", "./configs/rover_frame.json")
