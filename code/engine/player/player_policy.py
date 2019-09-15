import random
from time import sleep, time
from typing import Optional, Union, List

from engine.data import Positions
from engine.network import Network
from engine.player import Player


class PlayerPolicy(Player):
    def __init__(self, model_path: Optional[str] = None, min_wait_time: float = 0.0, max_wait_time: float = 0.0,
                 verbose: bool = False, gpus: Union[int, List[int]] = 0, max_gpu_mem_fraction: float = 1.0):
        super().__init__("LazyBug-Policy")
        self.max_wait_time = max_wait_time
        self.min_wait_time = min_wait_time
        self.__network = Network(model_path=model_path, gpus=gpus, max_gpu_mem_fraction=max_gpu_mem_fraction)
        # Perform first prediction to force keras to load the model
        self.__network.evaluate_positions(Positions.create_empty(1))
        self.verbose = verbose

    def move(self):
        start = time()
        min_move_time = self.min_wait_time + random.random() * (self.max_wait_time - self.min_wait_time)
        probs, value = self.__network.evaluate_game_view(game_view=self.game_view)
        if self.verbose:
            print("")
            probs_sorted = sorted(probs.items(), key=lambda i: i[1], reverse=True)
            for m, p in probs_sorted:
                print("{}: {}".format(m, p))
            print("Value: {}".format(value))
        move = max(probs.items(), key=lambda x: x[1], default=[None, None])[0]
        time_diff = min_move_time - (time() - start)
        if time_diff >= 0:
            sleep(time_diff)
        return move
