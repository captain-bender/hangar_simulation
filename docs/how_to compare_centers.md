## How to compare the centres coordinates OBB

In order to compare the centres, we need first to have a list of the inference coordinates from the model (e.g. YOLO obb) which are refered as "pred_centers" and the annotated ones which are refered as "gt_centers". The file that contains both:
```
centers_by_image.json
```
and it is generated from the test_evaluation.py script in the panther-obb repository.

Then in order to correlate the pixel coordinates with the actual cooridinates in the simulation, you need to execute the following script in Blender:
```
image2real_batch_conversion.Text.001
```
Do not forget to set the CENTERS_KEY variable to select the predictions or the annotated ones.

We need to check if there is any incosistency the the xy axis in the dataset. If we execute the following script:
```
python .\eucledian_distance_centres.py --list-swapped --swapped-out evaluation/v4/swapped_positions.json
```
we will get the list with positions that have the issues.

The in order to compare the centres with the ground truth you need to run:
```
python .\eucledian_distance_centres.py
```

Then for every category, pred and gt, generate a summary of statistics using the following script and updating the hardcoded paths accordingly.
```
python .\summarise_distance_errors.py
```

Finally, plot statistics related graphs using the following command:
```
python .\plot_error_statistics.py
```

## How to compare the centres coordinates Pose

In order to compare the centres, we need first to have a list of the inference coordinates from the model (e.g. YOLO pose) which are refered as "pred_centers" and the annotated ones which are refered as "gt_centers". The file that contains both:
```
apex_distances_debug.json
```
and it is generated from the test_evaluation.py script in the panther-pose repository.

The next step is to convert the format of this file to the one that is recognized in Blender using the follwowing script:
```
python convert_apex_to_centers_format.py "evaluation/v1-pose/apex_distances_debug.json" -o "evaluation/v1-pose/centers_by_image.json"
```

Then in order to correlate the pixel coordinates with the actual cooridinates in the simulation, you need to execute the following script in Blender:
```
image2real_batch_conversion.Text.001
```
Do not forget to set the CENTERS_KEY variable to select the predictions or the annotated ones.

We need to check if there is any incosistency the the xy axis in the dataset. If we execute the following script:
```
python .\eucledian_distance_centres_pose.py --list-swapped --swapped-out evaluation/v1-pose/swapped_positions.json
```
we will get the list with positions that have the issues.

The in order to compare the centres with the ground truth you need to run:
```
python .\eucledian_distance_centres_pose.py
```

Then for every category, pred and gt, generate a summary of statistics using the following script and updating the hardcoded paths accordingly.
```
python .\summarise_distance_errors.py
```

Finally, plot statistics related graphs using the following command:
```
python .\plot_error_statistics.py
```