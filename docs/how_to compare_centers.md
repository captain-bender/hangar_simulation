## How to compare the centres coordinates

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
python .\eucledian_distance_centres.py --list-swapped --swapped-out evaluation/swapped_positions.json
```
we will get the list with positions that have the issues.

The in order to compare the centres with the ground truth you need to run:
```
python .\eucledian_distance_centres.py
```