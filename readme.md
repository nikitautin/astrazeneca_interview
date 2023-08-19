# AstraZeneca interview programming task

This repository contains my solution to the interview task from AstraZeneca. It converts the output of the instance segmentation of individual cells on digital pathology slides into a semantic segmentation mask.

## Installation

Python 3.10+ is required.

1. ```pip install -r requirements.txt```
2. ```pip install -e .```

## Code structure

 - `convert_to_semantic_mask.py` It contains the `convert_to_semantic_mask` function that takes as input arguments the path to the original image and the path to the model outputs. Also, it has optional arguments for changing default threshold values. The function returns the semantic segmentation mask as a numpy array.
 - `convert_and_save_outputs.py` It's the script that converts model outputs for all images in a folder and saves them as semantic masks overlayed on top of the original images. To run it: `python convert_and_save_outputs.py --images_dir {IMAGES_DIR} --model_outputs_dir {MODEL_OUTPUT_DIR} --semantic_output_dir {SEMANTIC_OUTPUT_DIR}`
 - `detection_object.py` It contains the `DetectionObject` class that stores all information about single detection and provides useful functions. The class is used internally in `convert_to_semantic_mask.py`.
 - `tests/test_detection_object.py` It contains tests for DetectionObject class functionality.

 ## Algorithm

 Due to time constraints, I decided to go with a basic algorithm to resolving overlapping objects. Firstly, I check each pair of detected objects for intersection and find the object that has more total pixels probability in the intersection area than the other. After it, if another detected object intersects with the first object by more than the `merge_threshold` percentage, I merge its mask with that of the first object. Lastly, if objects weren't merged, then I exclude the intersected area from the other detected object. 

Additionally, I tried to enhance the algorithm by estimating an optimal ellipse for each detected object, but without ground truth semantic masks for the data, I decided not to go this way. Also, without ground truth labels, it’s challenging to select appropriate threshold values. The algorithm has only two thresholds: `binarization_threshold` that is used to convert raw masks to binary and `merge_threshold` as I mentioned before.



