from typing import Tuple

import numpy as np

import chess
import chess.variant
from engine.data import Positions, NUM_MOVE_IDS
from engine.player.evaluator import Evaluator


class EvaluatorSimple(Evaluator):
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 230,
        chess.BISHOP: 230,
        chess.ROOK: 250,
        chess.QUEEN: 450,
        chess.KING: 500
    }

    def evaluate_positions(self, positions: Positions) -> Tuple[np.ndarray, np.ndarray]:
        # count pieces of each team
        piece_values_np = np.array([self.PIECE_VALUES[p] for p in chess.PIECE_TYPES])
        piece_counts = np.sum(positions.pieces, axis=(-4, -2, -1))
        piece_counts_own = piece_counts[..., :6]
        piece_counts_other = piece_counts[..., 6:]

        team_pockets = np.sum(positions.pockets, axis=-3)
        team_pockets_own = team_pockets[..., 0, :]
        team_pockets_other = team_pockets[..., 1, :]

        piece_counts_own[..., :5] += team_pockets_own
        piece_counts_other[..., :5] += team_pockets_other

        values = piece_counts_own.dot(piece_values_np) - piece_counts_other.dot(piece_values_np)
        probabilities = np.ones((len(positions), NUM_MOVE_IDS)) / NUM_MOVE_IDS
        return probabilities, values
