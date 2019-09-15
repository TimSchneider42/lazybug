import random
from time import sleep

import chess
import chess.variant
from engine.game import OWN_BOARD
from engine.player import Player


class PlayerRandomAggressive(Player):
    def __init__(self, min_wait_time: float = 0.5, max_wait_time: float = 3.0):
        super().__init__("LazyBug-Random")
        self.max_wait_time = max_wait_time
        self.min_wait_time = min_wait_time

    def check_good(self, move: chess.Move, board: chess.variant.SingleBughouseBoard):
        if move.promotion or board.piece_at(move.to_square) is not None:
            return True
        p = board.piece_at(move.from_square)
        if p is not None:
            if p.piece_type == chess.PAWN:
                return True
        return False

    def move(self):
        board = self.game_view[OWN_BOARD]
        good_moves = [m for m in board.legal_moves if self.check_good(m, board)]
        move = random.choice(list(board.legal_moves) if len(good_moves) == 0 else good_moves)
        sleep(self.min_wait_time + random.random() * (self.max_wait_time - self.min_wait_time))
        return move