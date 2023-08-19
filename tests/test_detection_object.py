import pytest

import numpy as np

from astrazeneca_interview.detection_object import DetectionObject


@pytest.fixture
def detections_not_intersect_xy() -> tuple[DetectionObject, DetectionObject]:
    return (
        DetectionObject(0.7, np.array([0, 0, 10, 10]), np.full((28, 28), 1.0, dtype=np.float32)),
        DetectionObject(0.7, np.array([11, 20, 20, 28]), np.full((28, 28), 1.0, dtype=np.float32)),
    )


@pytest.fixture
def detections_not_intersect_x() -> tuple[DetectionObject, DetectionObject]:
    return (
        DetectionObject(0.7, np.array([0, 0, 10, 10]), np.full((28, 28), 1.0, dtype=np.float32)),
        DetectionObject(0.7, np.array([9, 11, 20, 28]), np.full((28, 28), 1.0, dtype=np.float32)),
    )


@pytest.fixture
def detections_not_intersect_y() -> tuple[DetectionObject, DetectionObject]:
    return (
        DetectionObject(0.7, np.array([0, 0, 10, 10]), np.full((28, 28), 1.0, dtype=np.float32)),
        DetectionObject(0.7, np.array([20, 10, 21, 28]), np.full((28, 28), 1.0, dtype=np.float32)),
    )


@pytest.fixture
def detections_intersect_only_box() -> tuple[DetectionObject, DetectionObject]:
    return (
        DetectionObject(0.7, np.array([5, 0, 10, 10]), np.full((28, 28), 0.0, dtype=np.float32)),
        DetectionObject(0.7, np.array([0, 10, 21, 28]), np.full((28, 28), 0.0, dtype=np.float32)),
    )


@pytest.fixture
def detections_intersect_fully() -> tuple[DetectionObject, DetectionObject]:
    return (
        DetectionObject(0.7, np.array([5, 0, 10, 10]), np.full((28, 28), 1.0, dtype=np.float32)),
        DetectionObject(0.7, np.array([0, 10, 21, 28]), np.full((28, 28), 1.0, dtype=np.float32)),
    )


def test_detection_init():
    raw_maskrcnn_mask = np.zeros((28, 28), dtype=np.float32)
    raw_maskrcnn_mask[:14, :14] = 1.0
    raw_maskrcnn_mask[7:10, 2:3] = 0.0
    det = DetectionObject(score=0.9, box=np.array([100, 200, 155, 227]), mask=raw_maskrcnn_mask)

    assert det.score == 0.9
    assert det.raw_mask.shape == (56, 28)
    assert (
        det.binary_mask.shape == (56, 28)
        and det.binary_mask.sum() == 28 * 14
        and (det.binary_mask[:28, :14] == 1).all()
    )


@pytest.mark.parametrize(
    "detections",
    ["detections_not_intersect_xy", "detections_not_intersect_x", "detections_not_intersect_y"],
)
def test_box_not_intersect(detections, request):
    det1, det2 = request.getfixturevalue(detections)
    assert not det1.box_intersects(det2)


@pytest.mark.parametrize(
    "detections",
    ["detections_intersect_only_box", "detections_intersect_fully"],
)
def test_box_intersect(detections, request):
    det1, det2 = request.getfixturevalue(detections)
    assert det1.box_intersects(det2)


@pytest.mark.parametrize(
    "detections",
    [
        "detections_not_intersect_xy",
        "detections_not_intersect_x",
        "detections_not_intersect_y",
        "detections_intersect_only_box",
        "detections_intersect_fully",
    ],
)
def test_box_intersect_commutative(detections, request):
    det1, det2 = request.getfixturevalue(detections)
    assert det1.box_intersects(det2) == det2.box_intersects(det1)


@pytest.mark.parametrize(
    "detections",
    [
        "detections_intersect_only_box",
        "detections_not_intersect_xy",
        "detections_not_intersect_x",
        "detections_not_intersect_y",
    ],
)
def test_mask_not_intersect(detections, request):
    det1, det2 = request.getfixturevalue(detections)
    assert not det1.mask_intersects(det2)


def test_mask_intersect(detections_intersect_fully):
    det1, det2 = detections_intersect_fully
    assert det1.mask_intersects(det2)


@pytest.mark.parametrize(
    "detections",
    [
        "detections_not_intersect_xy",
        "detections_not_intersect_x",
        "detections_not_intersect_y",
        "detections_intersect_only_box",
        "detections_intersect_fully",
    ],
)
def test_mask_intersect_commutative(detections, request):
    det1, det2 = request.getfixturevalue(detections)
    assert det1.mask_intersects(det2) == det2.mask_intersects(det1)


def test_paste_and_crop_mask():
    det = DetectionObject(0.9, np.array([100, 20, 120, 58]), np.full((28, 28), 1.0, dtype=np.float32))
    mask_before = det.binary_mask.copy()
    origin_xy = (0, 0)
    image = det.paste_mask_to_image(image_size_wh=(1000, 500), origin_xy=origin_xy)
    det._crop_mask_to_bbox(image, origin_xy=origin_xy)
    np.testing.assert_array_equal(mask_before, det.binary_mask)


def test_fix_intersection_detections_not_intersets():
    det1 = DetectionObject(0.9, np.array([0, 0, 10, 10]), np.full((28, 28), 1.0, dtype=np.float32))
    det2 = DetectionObject(0.7, np.array([11, 20, 20, 28]), np.full((28, 28), 0.8, dtype=np.float32))
    mask1_before = det1.binary_mask.copy()
    mask2_before = det2.binary_mask.copy()

    det1.fix_intersection(det2)
    np.testing.assert_array_equal(mask1_before, det1.binary_mask)
    np.testing.assert_array_equal(mask2_before, det2.binary_mask)

    det2.fix_intersection(det1)
    np.testing.assert_array_equal(mask1_before, det1.binary_mask)
    np.testing.assert_array_equal(mask2_before, det2.binary_mask)


def test_fix_intersection_detections_intersets():
    det1 = DetectionObject(0.9, np.array([0, 0, 20, 20]), np.full((28, 28), 1.0, dtype=np.float32))
    det2 = DetectionObject(0.7, np.array([10, 5, 30, 45]), np.full((28, 28), 0.8, dtype=np.float32))
    mask1_before = det1.binary_mask.copy()
    mask2_area_before = det2.binary_mask.sum()

    det1.fix_intersection(det2)
    assert not det1.is_merged and not det2.is_merged
    np.testing.assert_array_equal(mask1_before, det1.binary_mask)
    assert det2.binary_mask.sum() + 11 * 16 == mask2_area_before
    assert (det2.binary_mask[:11, :16] == 0).all()


def test_fix_intersection_detection_fully_inside():
    det1 = DetectionObject(0.9, np.array([0, 0, 20, 20]), np.full((28, 28), 1.0, dtype=np.float32))
    det2 = DetectionObject(0.7, np.array([10, 15, 15, 17]), np.full((28, 28), 0.8, dtype=np.float32))
    mask1_before = det1.binary_mask.copy()

    det1.fix_intersection(det2)
    assert not det1.is_merged and det2.is_merged
    np.testing.assert_array_equal(mask1_before, det1.binary_mask)


def test_fix_intersection_merge_detections():
    det1 = DetectionObject(0.7, np.array([10, 20, 20, 40]), np.full((28, 28), 0.6, dtype=np.float32))
    det2 = DetectionObject(0.9, np.array([0, 0, 20, 30]), np.full((28, 28), 0.8, dtype=np.float32))

    det1.fix_intersection(det2)
    assert det1.is_merged and not det2.is_merged
    assert det2.binary_mask.shape == (21, 41)
    assert (det2.binary_mask[:10, 31:40] == 0).all()
    assert det2.binary_mask.sum() + 100 == 21 * 41
