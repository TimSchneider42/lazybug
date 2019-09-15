import math
import time
from typing import Optional, Tuple
import numpy as np

import chess
from chess.variant import BughouseBoards, LEFT, RIGHT

PLAYER_NAMES = [["a", "A"], ["b", "B"]]


class BughouseGame(BughouseBoards):
    def __init__(self, clock_time_limit_s: float = 120, clock_increment_s: float = 0, fen: Optional[str] = None):
        super().__init__(fen=fen, chess960=False)
        self.__clock_time_limit_s = clock_time_limit_s
        self.__clock_increment_s = clock_increment_s
        self.__game_start_time: Optional[float] = None
        self.__player_names = [[n for n in b] for b in PLAYER_NAMES]
        self.__final_clocks: Optional[np.ndarray] = None
        self.flip_visualization = False

    def __check_game_started(self):
        assert self.game_started, "The game has not yet been started."

    def __check_game_not_started(self):
        assert not self.game_started, "The game has already been started."

    def start_clocks(self, time_s: Optional[float] = None):
        if time_s is None:
            time_s = time.time()
        self.__check_game_not_started()
        self.__game_start_time = time_s

    def push(self, move: chess.Move, time_s: Optional[float] = None, check_game_over: bool = True):
        if time_s is None:
            time_s = time.time()
        self.__check_game_started()
        if move.move_time is None:
            move.move_time = self.get_clocks_s(time_s=time_s)[move.board_id][int(self[move.board_id].turn)]
        super().push(move)
        if check_game_over:
            if self.is_game_over(time_s=time_s, check_clocks=False):
                self.__final_clocks = self.get_clocks_s(time_s=time_s)

    def reset(self):
        super().reset()
        self.__game_start_time = None
        self.__final_clocks = None

    @property
    def clock_time_limit_s(self) -> float:
        return self.__clock_time_limit_s

    @clock_time_limit_s.setter
    def clock_time_limit_s(self, value: float):
        self.__check_game_not_started()
        self.__clock_time_limit_s = value

    @property
    def clock_increment_s(self) -> float:
        return self.__clock_increment_s

    @clock_increment_s.setter
    def clock_increment_s(self, value: float):
        self.__check_game_not_started()
        self.__clock_increment_s = value

    @property
    def clocks_s(self) -> np.ndarray:
        return self.get_clocks_s()

    @property
    def game_started(self) -> bool:
        return self.__game_start_time is not None

    def is_game_over(self, time_s: Optional[float] = None, check_clocks: bool = True) -> bool:
        return super().is_game_over() or (self.is_time_up(time_s=time_s) and check_clocks)

    def is_time_up(self, time_s: Optional[float] = None) -> bool:
        return np.any(self.get_clocks_s(time_s=time_s) <= 0)

    def get_clock_s(self, board_id: int, time_s: Optional[float] = None):
        if not self.game_started:
            return np.array([self.clock_time_limit_s, self.clock_time_limit_s])
        if time_s is None:
            time_s = time.time()
        clocks_s = np.zeros(2)
        time_since_start = time_s - self.__game_start_time
        b = self[board_id]
        ms = self[board_id].move_stack
        other_player_clock = self.__clock_time_limit_s if len(ms) == 0 else ms[-1].move_time
        other_player_move_count = math.ceil(len(ms) / 2)
        other_player_time_spent = self.__clock_time_limit_s - other_player_clock + \
                                  other_player_move_count * self.clock_increment_s
        moving_player_move_count = len(ms) // 2
        moving_player_time_spent = time_since_start - other_player_time_spent
        moving_player_clock = self.clock_time_limit_s - moving_player_time_spent + \
                              moving_player_move_count * self.clock_increment_s
        clocks_s[int(b.turn)] = moving_player_clock
        clocks_s[int(not b.turn)] = other_player_clock
        return clocks_s

    def get_clocks_s(self, time_s: Optional[float] = None):
        if time_s is None:
            time_s = time.time()
        return np.array([self.get_clock_s(i, time_s) for i in range(len(self))])

    def result(self, time_s: Optional[float] = None) -> str:
        result = super().result()
        if result == "*":
            clocks_s = self.get_clocks_s(time_s=time_s)
            min_clock_index = np.unravel_index(np.argmin(clocks_s), clocks_s.shape)
            if clocks_s[min_clock_index] <= 0:
                board, color = min_clock_index
                result = "0-1" if (board == chess.variant.BOARD_A) == (color == chess.WHITE) else "1-0"
            elif not (any(True for _ in self.boards[LEFT].legal_moves) or
                      any(True for _ in self.boards[RIGHT].legal_moves)):
                clocks_s = self.get_clocks_s(time_s=time_s)
                loser_board = int(np.argmin([clocks_s[b][int(self[b].turn)] for b in chess.variant.BOARDS]))
                losing_player = self[loser_board].turn
                return "0-1" if (loser_board == chess.variant.BOARD_A) == (losing_player == chess.WHITE) else "1-0"
        return result

    def result_comment(self, time_s: Optional[float] = None):
        if self.is_game_over():
            if self.is_threefold_repetition():
                return "Game drawn by threefold repetition"
            elif self.is_checkmate():
                for i, b in enumerate(self):
                    if b.is_checkmate():
                        losing_player = b.turn
                        return "{} ({}) checkmated".format(
                            PLAYER_NAMES[i][losing_player], self.__player_names[i][losing_player])
            elif self.is_time_up(time_s=time_s):
                clocks_s = self.get_clocks_s(time_s=time_s)
                b, c = np.unravel_index(np.argmin(clocks_s), clocks_s.shape)
                return "Player {} ({}) forfeits on time".format(PLAYER_NAMES[b][c], self.__player_names[b][c])
            elif not (any(True for _ in self.boards[LEFT].legal_moves) or
                      any(True for _ in self.boards[RIGHT].legal_moves)):
                clocks_s = self.get_clocks_s(time_s=time_s)
                loser_board = int(np.argmin([clocks_s[b][int(self[b].turn)] for b in chess.variant.BOARDS]))
                losing_player = self[loser_board].turn
                return "{} ({}) cannot move and will run out of time.".format(
                    PLAYER_NAMES[loser_board][losing_player], self.__player_names[loser_board][losing_player])
            else:
                return "Game aborted"
        return "Game still running"

    @property
    def player_names(self) -> Tuple[Tuple[str, ...], ...]:
        return tuple(tuple(n) for n in self.__player_names)

    def set_player_name(self, board_id: int, color: bool, name: str):
        self.__player_names[board_id][color] = name

    def _repr_svg_(self):
        import chess.svg
        clocks = self.get_final_clocks_s()
        if clocks is None:
            clocks = self.clocks_s
        player_text = [["{} - {:0.1f}s".format(n, t) for n, t in zip(bn, bt)] for bn, bt in
                       zip(self.__player_names, clocks)]
        return chess.svg.bughouse_boards(
            boards=self,
            size=800,
            flipped=self.flip_visualization,
            lastmoveL=self.boards[LEFT].peek() if self.boards[LEFT].move_stack else None,
            lastmoveR=self.boards[RIGHT].peek() if self.boards[RIGHT].move_stack else None,
            checkL=self.boards[LEFT].king(self.boards[LEFT].turn) if self.boards[LEFT].is_check() else None,
            checkR=self.boards[RIGHT].king(self.boards[RIGHT].turn) if self.boards[RIGHT].is_check() else None,
            player_text=player_text
        )

    def get_final_clocks_s(self, time_s: Optional[float] = None) -> Optional[np.ndarray]:
        if self.__final_clocks is not None:
            return self.__final_clocks
        clocks = self.get_clocks_s(time_s=time_s)
        if np.any(clocks <= 0):
            min_clock = np.min(clocks)
            for i, b in enumerate(self):
                clocks[i][int(b.turn)] += min_clock
            return clocks
        return None
