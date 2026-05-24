from src.evaluation.metrics import angle_diff_deg, iou_xyxy


def test_iou_xyxy() -> None:
    assert round(iou_xyxy((0, 0, 10, 10), (5, 5, 15, 15)), 3) == 0.143


def test_angle_diff_deg() -> None:
    assert angle_diff_deg(2, 178) == 4

