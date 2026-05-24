from pathlib import Path

from src.data.validate_dataset import validate_dataset


def test_validate_dataset(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "a.jpg").write_bytes(b"x")
    (images / "b.jpg").write_bytes(b"x")
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
    result = validate_dataset(images, labels)
    assert result.total_images == 2
    assert result.labeled_images == 1
    assert result.missing_labels == 1

