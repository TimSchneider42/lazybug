from typing import Optional

from engine.data import Positions
from engine.game import OWN_BOARD
from engine.player import Player
from engine.player.evaluator_network import EvaluatorNetwork
from engine.player.evaluator_simple import EvaluatorSimple
from engine.player.mcts import MCTS

from engine.player.mcts_node import WAIT_MOVE


class PlayerMCTS(Player):
    def __init__(self, model_path: Optional[str] = None, min_move_time_s: float = 5.0, max_move_time_s: float = 10.0,
                 max_gpu_mem_fraction: Optional[float] = None, logging_directory: Optional[str] = None,
                 initial_random_moves: int = 0, parallel_evaluations: int = 8):
        super().__init__("LazyBug-MCTS")
        self.max_move_time_s = max_move_time_s
        if model_path is not None:
            evaluator = EvaluatorNetwork(model_path, max_gpu_mem_fraction)
            # Perform first prediction to force keras to load the model
            evaluator.evaluate_positions(Positions.create_empty(parallel_evaluations))
        else:
            evaluator = EvaluatorSimple()
        self._mcts = MCTS(evaluator, parallel_evaluations=parallel_evaluations, logging_directory=logging_directory)
        self.__initial_random_moves = initial_random_moves
        self.__min_move_time_s = min_move_time_s
        self.__previous_partner_halfmove_clock = 0

    def __on_recv_ptell(self, sender, msg: str):
        cmd = msg.strip().split(" ")[0]
        if cmd == "sitting":
            self._mcts.partner_is_sitting = True
        elif cmd == "going":
            self._mcts.partner_is_sitting = False

    def __on_game_state_changed(self, sender, args):
        if self.game_view[int(not OWN_BOARD)].halfmove_clock != self.__previous_partner_halfmove_clock:
            self._mcts.partner_is_sitting = False
            self.__previous_partner_halfmove_clock = self.game_view[int(not OWN_BOARD)].halfmove_clock

    def on_game_started(self):
        self.__previous_partner_halfmove_clock = 0
        self.game_view.recv_ptell_event.add(self.__on_recv_ptell)
        self.game_view.game_state_changed_event.add(self.__on_game_state_changed)
        self._mcts.start_new_game(self.game_view)

    def move(self):
        own_clock = self.game_view.clocks_s[OWN_BOARD, int(self.game_view.own_color)]
        max_move_time = min(
            self.max_move_time_s,
            own_clock - 20.0,
            max(0.1 * (own_clock - 20.0), self.__min_move_time_s))
        play_random = self.game_view[OWN_BOARD].fullmove_number <= self.__initial_random_moves
        move = self._mcts.compute_next_move(
            min_time_s=self.__min_move_time_s, max_time_s=max_move_time, play_random=play_random)
        if move is not WAIT_MOVE:
            return move
        else:
            self.game_view.send_ptell("sitting ({})".format(self._mcts.explanation))
            print("sitting ({})".format(self._mcts.explanation))
            return None
