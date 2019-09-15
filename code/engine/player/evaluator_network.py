from typing import Tuple, Optional
import numpy as np

from engine.data import Positions
from engine.network import Network
from engine.player.evaluator import Evaluator


class EvaluatorNetwork(Evaluator):
    def __init__(self, model_path: str, max_gpu_mem_fraction: Optional[float] = None):
        self.__network = Network(model_path=model_path, max_gpu_mem_fraction=max_gpu_mem_fraction)

    def evaluate_positions(self, positions: Positions) -> Tuple[np.ndarray, np.ndarray]:
        return self.__network.evaluate_positions(positions)