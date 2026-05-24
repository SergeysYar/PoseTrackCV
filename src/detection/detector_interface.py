from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.pose.pose_estimator import PosePrediction


class Detector(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> list[PosePrediction]:
        raise NotImplementedError

    @abstractmethod
    def predict_path(self, image_path: Path) -> list[PosePrediction]:
        raise NotImplementedError

