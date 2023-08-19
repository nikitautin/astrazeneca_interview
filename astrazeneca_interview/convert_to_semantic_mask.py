from pathlib import Path

import numpy as np
from skimage.io import imread

from astrazeneca_interview.detection_object import DetectionObject


def read_model_output(output_name: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(output_name) as data:
        boxes = data["boxes"]
        class_ids = data["class_ids"]
        assert (class_ids == class_ids[0]).all()
        scores = data["scores"]
        masks = data["masks"]
    return boxes, scores, masks


def create_detections_list(
    boxes: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    binarization_threshold: float,
    merge_threshold: float,
) -> list[DetectionObject]:
    return [
        DetectionObject(
            score=score,
            box=box,
            mask=mask,
            binarization_threshold=binarization_threshold,
            merge_threshold=merge_threshold,
        )
        for score, box, mask in zip(scores, boxes, masks)
    ]


def fix_intersections(detections: list[DetectionObject]) -> None:
    for i, main_detection in enumerate(detections):
        for other_detection in detections[i + 1 :]:
            main_detection.fix_intersection(other_detection)


def make_semantic_mask(detections: list[DetectionObject], size_wh: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(size_wh, dtype=np.uint32)
    for i, detection in enumerate(detections):
        if detection.is_merged:
            continue
        mask = detection.paste_mask_to_image(size_wh, origin_xy=(0, 0), image=mask, mask_value=i + 1)
        assert mask.max() == i + 1
    return mask


def convert_to_semantic_mask(
    image_path: Path,
    model_output_path: Path,
    binarization_threshold: float = 0.5,
    merge_threshold: float = 0.5,
    return_image: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    image = imread(image_path)

    boxes, scores, masks = read_model_output(model_output_path)
    detections = create_detections_list(boxes, scores, masks, binarization_threshold, merge_threshold)
    fix_intersections(detections)
    semantic_mask = make_semantic_mask(detections, image.shape[:2])

    if return_image:
        return semantic_mask, image
    else:
        return semantic_mask
