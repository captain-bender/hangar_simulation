#!/usr/bin/env python3
"""
Convert apex_distances_debug.json format to centers_by_image.json format.
"""

import json
import argparse
from pathlib import Path


def convert_apex_to_centers_format(input_file, output_file=None, include_metrics=False):
    """
    Convert apex distances format to centers by image format.
    
    Args:
        input_file: Path to apex_distances_debug.json
        output_file: Path to output file (defaults to input_file.converted.json)
        include_metrics: Whether to include distance_px and bbox_iou in output
    """
    
    # Load input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract apex distances array
    apex_distances = data.get('apex_distances', [])
    
    # Convert to centers format
    converted = []
    
    for entry in apex_distances:
        converted_entry = {
            "file": entry["image"],
            "pred_centers": [
                {
                    "cls": 0,
                    "x": entry["pred_apex"][0],
                    "y": entry["pred_apex"][1],
                    "conf": entry["bbox_iou"]
                }
            ],
            "gt_centers": [
                {
                    "cls": 0,
                    "x": entry["gt_apex"][0],
                    "y": entry["gt_apex"][1]
                }
            ]
        }
        
        # Optionally include the distance and IOU metrics
        if include_metrics:
            converted_entry["distance_px"] = entry["distance_px"]
            converted_entry["bbox_iou"] = entry["bbox_iou"]
        
        converted.append(converted_entry)
    
    # Determine output file
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}.converted.json"
    
    # Write output file
    with open(output_file, 'w') as f:
        json.dump(converted, f, indent=2)
    
    print(f"✓ Converted {len(converted)} entries")
    print(f"✓ Output saved to: {output_file}")
    
    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert apex_distances_debug.json to centers_by_image.json format"
    )
    parser.add_argument(
        "input_file",
        help="Path to apex_distances_debug.json file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output file (optional)"
    )
    parser.add_argument(
        "--include-metrics",
        action="store_true",
        help="Include distance_px and bbox_iou in output"
    )
    
    args = parser.parse_args()
    
    convert_apex_to_centers_format(
        args.input_file,
        args.output,
        args.include_metrics
    )
