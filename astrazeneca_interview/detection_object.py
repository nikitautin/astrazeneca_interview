import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.transform import resize


class DetectionObject:
    def __init__(
        self,
        score: float,
        box: np.ndarray,
        mask: np.ndarray,
        binarization_threshold: float = 0.5,
        merge_threshold: float = 0.5,
    ):
        self.score = score
        self.min_x, self.min_y, self.max_x, self.max_y = box
        self.raw_mask = self._resize_mask(mask)
        self.binary_mask = self.raw_mask > binarization_threshold
        self.binary_mask = binary_fill_holes(self.binary_mask).astype(np.uint8)

        self.merge_threshold = merge_threshold
        self.is_merged = False

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        w = max(self.max_x - self.min_x + 1, 1)
        h = max(self.max_y - self.min_y + 1, 1)
        return resize(mask, (w, h), order=1)

    def intersects(self, other: "DetectionObject") -> bool:
        return self.box_intersects(other) and self.mask_intersects(other)

    def box_intersects(self, other: "DetectionObject") -> bool:
        x_max_min = max(self.min_x, other.min_x)
        y_max_min = max(self.min_y, other.min_y)
        x_min_max = min(self.max_x, other.max_x)
        y_min_max = min(self.max_y, other.max_y)
        return x_min_max >= x_max_min and y_min_max >= y_max_min

    def _get_merged_masks_stats(self, other: "DetectionObject") -> tuple[tuple[int, int], tuple[int, int]]:
        origin_x = min(self.min_x, other.min_x)
        origin_y = min(self.min_y, other.min_y)

        max_x = max(self.max_x, other.max_x)
        max_y = max(self.max_y, other.max_y)

        w = max_x - origin_x + 1
        h = max_y - origin_y + 1
        return (origin_x, origin_y), (w, h)

    def mask_intersects(self, other: "DetectionObject") -> bool:
        origin_xy, size_wh = self._get_merged_masks_stats(other)
        self_mask = self.paste_mask_to_image(size_wh, origin_xy)
        other_mask = other.paste_mask_to_image(size_wh, origin_xy)
        return (self_mask * other_mask).sum() > 0

    def fix_intersection(self, other: "DetectionObject") -> None:
        if not self.intersects(other) or self.is_merged or other.is_merged:
            return
        origin_xy, size_wh = self._get_merged_masks_stats(other)
        self_binary_mask = self.paste_mask_to_image(size_wh, origin_xy)
        other_binary_mask = other.paste_mask_to_image(size_wh, origin_xy)

        self_raw_mask = self.paste_mask_to_image(size_wh, origin_xy, self.raw_mask)
        other_raw_mask = other.paste_mask_to_image(size_wh, origin_xy, other.raw_mask)

        intersection_mask = other_binary_mask * self_binary_mask
        self_intersection_score = (self_raw_mask * intersection_mask).sum()
        other_intersection_score = (other_raw_mask * intersection_mask).sum()
        if self_intersection_score > other_intersection_score:
            self._fix_intersection(other, self_binary_mask, other_binary_mask, intersection_mask, origin_xy)
        else:
            other._fix_intersection(self, other_binary_mask, self_binary_mask, intersection_mask, origin_xy)

    def _fix_intersection(
        self,
        other: "DetectionObject",
        self_binary_mask: np.ndarray,
        other_binary_mask: np.ndarray,
        intersection_mask: np.ndarray,
        origin_xy: tuple[int, int],
    ) -> None:
        if self.merge_threshold * other.binary_mask.sum() < intersection_mask.sum():
            self._merge_detections(other, self_binary_mask, other_binary_mask, origin_xy)
        else:
            other._exclude_intersected_area(other_binary_mask, intersection_mask, origin_xy)

    def _merge_detections(
        self,
        other: "DetectionObject",
        self_binary_mask: np.ndarray,
        other_binary_mask: np.ndarray,
        origin_xy: tuple[int, int],
    ) -> None:
        merged_mask = self_binary_mask | other_binary_mask
        self._crop_mask_to_bbox(
            merged_mask,
            origin_xy,
            bbox=self._get_common_bbox(other)
        )
        other.is_merged = True

    def _get_common_bbox(self, other: "DetectionObject") -> np.ndarray:
        return np.array([
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
        ])

    def _exclude_intersected_area(
        self, self_binary_mask: np.ndarray, intersected_area: np.ndarray, origin_xy: tuple[int, int]
    ) -> None:
        excluded_mask = self_binary_mask - intersected_area
        self._crop_mask_to_bbox(excluded_mask, origin_xy)

    def _crop_mask_to_bbox(
        self,
        mask: np.ndarray,
        origin_xy: tuple[int, int],
        bbox: np.ndarray | None = None,
    ):
        if bbox is not None:
            self.min_x, self.min_y, self.max_x, self.max_y = bbox

        mask_w = self.max_x - self.min_x + 1
        mask_h = self.max_y - self.min_y + 1
        offset_x = self.min_x - origin_xy[0]
        offset_y = self.min_y - origin_xy[1]
        self.binary_mask = mask[offset_x : offset_x + mask_w, offset_y : offset_y + mask_h]

    def paste_mask_to_image(
        self,
        image_size_wh: tuple[int, int],
        origin_xy: tuple[int, int],
        mask: np.ndarray | None = None,
        image: np.ndarray | None = None,
        mask_value: int = 1,
    ) -> np.ndarray:
        if image is None:
            image = np.zeros(image_size_wh, dtype=np.uint8 if mask is None else mask.dtype)

        offset_xy = np.array([self.min_x, self.min_y]) - np.array(origin_xy)
        mask_wh = np.array(self.binary_mask.shape[:2])
        mask_wh -= (offset_xy + mask_wh - np.array(image_size_wh)).clip(min=0)

        original_mask = self.binary_mask if mask is None else mask
        cropped_mask = (original_mask[:mask_wh[0], :mask_wh[1]] * mask_value).astype(image.dtype)
        image[offset_xy[0] : offset_xy[0] + mask_wh[0], offset_xy[1] : offset_xy[1] + mask_wh[1]] += cropped_mask
        return image
