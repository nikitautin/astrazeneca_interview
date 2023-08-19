import argparse
from functools import partial
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map
from skimage.color import label2rgb
from skimage.io import imsave

from astrazeneca_interview.convert_to_semantic_mask import convert_to_semantic_mask


def convert_and_save(
    image_path: Path,
    model_outputs_dir: Path,
    semantic_output_dir: Path,
    binarization_threshold: float,
    merge_threshold: float,
) -> None:
    model_output_path = model_outputs_dir / f"{image_path.stem}.npz"
    semantic_mask, image = convert_to_semantic_mask(
        image_path,
        model_output_path,
        binarization_threshold=binarization_threshold,
        merge_threshold=merge_threshold,
        return_image=True,
    )

    image_with_segmentation = label2rgb(label=semantic_mask, image=image, saturation=1.0, alpha=0.4)
    imsave(semantic_output_dir / f"{image_path.stem}.png", (image_with_segmentation * 255).astype(np.uint8))


def convert_all_outputs_to_semantic(
    images_dir: Path,
    model_outputs_dir: Path,
    semantic_output_dir: Path,
    binarization_threshold: float = 0.5,
    merge_threshold: float = 0.5,
) -> None:
    semantic_output_dir.mkdir(parents=True, exist_ok=True)

    images = list(images_dir.glob("*.png"))
    save_func = partial(
        convert_and_save,
        model_outputs_dir=model_outputs_dir,
        semantic_output_dir=semantic_output_dir,
        binarization_threshold=binarization_threshold,
        merge_threshold=merge_threshold,
    )
    process_map(save_func, images, desc="Converting to semantic masks")


def main():
    parser = argparse.ArgumentParser(description="Convert and save all models outputs as semantic masks overlayed on top of the original images")
    parser.add_argument("--images_dir", type=Path, required=True, help="Directory containing images")
    parser.add_argument("--model_outputs_dir", type=Path, required=True, help="Directory containing model outputs")
    parser.add_argument("--semantic_output_dir", type=Path, required=True, help="Directory for saving semantic masks")
    parser.add_argument(
        "--binarization_threshold", type=float, default=0.5, help="Masks binarization threshold (default: 0.5)"
    )
    parser.add_argument("--merge_threshold", type=float, default=0.5, help="Masks merge threshold (default: 0.5)")

    args = parser.parse_args()
    convert_all_outputs_to_semantic(
        args.images_dir,
        args.model_outputs_dir,
        args.semantic_output_dir,
        args.binarization_threshold,
        args.merge_threshold,
    )


if __name__ == "__main__":
    main()
