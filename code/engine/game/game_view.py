import time
from abc import abstractmethod
from enum import Enum
from typing import Optional, Tuple

import chess.variant
import chess
from chess.variant import LEFT, RIGHT
from engine.game.bughouse_game import PLAYER_NAMES
from .bughouse_game import BughouseGame
from engine.util import Event

OWN_BOARD = chess.variant.BOARD_A
OTHER_BOARD = chess.variant.BOARD_B


class GameView(BughouseGame):
    def __init__(self, own_color: bool, clock_time_limit_s: float = 120, clock_increment_s: float = 0,
                 fen: Optional[str] = None, push_own_moves: bool = True):
        super().__init__(clock_time_limit_s=clock_time_limit_s, clock_increment_s=clock_increment_s, fen=fen)
        self.__own_color = own_color
        self.__game_state_changed_event = Event()
        self.__own_color_changed_event = Event()
        self.__recv_ptell_event = Event()
        self.__boards_reset_event = Event()
        self.__simulated_moves = [0, 0]
        self.__push_own_moves = push_own_moves
        self.__game_finished = False
        self.__waiting_for_own_move = False
        self.flip_visualization = self.__own_color == chess.BLACK

    def _push(self, move: chess.Move, time_s: Optional[float] = None):
        assert self.total_simulated_moves == 0, "Cannot push move while there are simulated moves on the move stack"
        super().push(move, time_s=time_s)
        if move.board_id == OWN_BOARD:
            self.__waiting_for_own_move = False
        self.__game_state_changed_event.fire(self)

    def push(self, move: chess.Move, time_s: Optional[float] = None):
        if time_s is None:
            time_s = time.time()
        assert self.total_simulated_moves == 0, "Cannot push move while there are simulated moves on the move stack"
        if move.board_id is None:
            move.board_id = OWN_BOARD
        assert move.board_id == OWN_BOARD
        assert self.boards[OWN_BOARD].turn == self.__own_color
        assert not self.is_game_over(time_s, check_clocks=False)
        self._send_move(move)
        if self.__push_own_moves:
            self._push(move, time_s=time_s)
        else:
            self.__waiting_for_own_move = True
            while self.__waiting_for_own_move and not self.game_finished(time_s=time_s):
                self.receive_updates()

    def push_simulation(self, move: chess.Move, time_s: Optional[float] = None):
        self.__simulated_moves[move.board_id] += 1
        super().push(move, time_s=time_s, check_game_over=False)

    def pop(self, board_index: Optional[int] = None) -> chess.Move:
        assert self.total_simulated_moves > 0, "No more simulated moves to pop."
        if board_index is None:
            move = super().pop()
        else:
            assert self.__simulated_moves[board_index] > 0
            move = super().pop(board_index)
        self.__simulated_moves[move.board_id] -= 1
        return move

    def _set_own_color(self, new_color: bool):
        if new_color != self.__own_color:
            name = self.player_names[OWN_BOARD][self.__own_color]
            self.set_player_name(OWN_BOARD, self.__own_color, PLAYER_NAMES[OWN_BOARD][self.__own_color])
            self.__own_color = new_color
            self.set_player_name(OWN_BOARD, self.__own_color, name)
            self.flip_visualization = self.__own_color == chess.BLACK
            self.__own_color_changed_event.fire(self)

    def _finish_game(self):
        self.__game_finished = True
        self.__waiting_for_own_move = False
        self.__game_state_changed_event.fire(self)

    def _reset(self):
        self.reset()
        self.__boards_reset_event.fire(self)
        self.__game_state_changed_event.fire(self)
        self.__simulated_moves = [0, 0]
        self.__game_finished = False
        self.__waiting_for_own_move = False
        raise GameAborted(GameAbortReasons.BOARD_RESET)

    def clear_simulated_moves(self):
        while self.total_simulated_moves > 0:
            self.pop()

    def _add_ptell_message(self, msg: str):
        self.__recv_ptell_event.fire(self, msg)

    @abstractmethod
    def _send_move(self, move: chess.Move):
        pass

    @abstractmethod
    def send_ptell(self, msg: str):
        pass

    @abstractmethod
    def updates_available(self) -> bool:
        pass

    @abstractmethod
    def receive_updates(self, timeout: Optional[float] = None) -> chess.Move:
        pass

    @property
    def own_color(self) -> bool:
        return self.__own_color

    @property
    def game_state_changed_event(self) -> Event:
        return self.__game_state_changed_event

    @property
    def own_color_changed_event(self) -> Event:
        return self.__own_color_changed_event

    @property
    def boards_reset_event(self) -> Event:
        return self.__boards_reset_event

    @property
    def simulated_moves(self) -> Tuple[int, int]:
        return tuple(self.__simulated_moves)

    @property
    def total_simulated_moves(self) -> int:
        return sum(self.__simulated_moves)

    @property
    def recv_ptell_event(self) -> Event:
        return self.__recv_ptell_event

    def game_finished(self, time_s: Optional[float] = None):
        return self.__game_finished or self.is_game_over(time_s=time_s, check_clocks=False)

    @abstractmethod
    def set_own_name(self, name: str):
        pass


class GameAbortReasons(Enum):
    BOARD_RESET = 0
    QUIT_SIGNAL = 1


class GameAborted(Exception):
    def __init__(self, reason: GameAbortReasons):
        self.__reason = reason

    @property
    def reason(self) -> GameAbortReasons:
        return self.__reason

    def __repr__(self) -> str:
        return "GameAborted({})".format(self.__reason.name)
