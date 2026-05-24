from src.pose.geometry_utils import bbox_center, normalize_angle_180


def test_bbox_center() -> None:
    assert bbox_center(0, 0, 10, 20) == (5.0, 10.0)


def test_normalize_angle_180() -> None:
    assert normalize_angle_180(190) == 10
    assert normalize_angle_180(-10) == 170

