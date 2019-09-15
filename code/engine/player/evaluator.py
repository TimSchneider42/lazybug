from abc import abstractmethod, ABC
from typing import Tuple
import numpy as np
from engine.data import Positions


class Evaluator(ABC):
    @abstractmethod
    def evaluate_positions(self, positions: Positions) -> Tuple[np.ndarray, np.ndarray]:
        pass
