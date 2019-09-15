import unittest

import chess.pgn
import chess.variant
from chess.variant import BughouseBoards
from engine import data


class TestMoveTransform(unittest.TestCase):
    def test_fromfile(self):
        path = '../../../data_bpgn/test.bpgn'
        with open(path) as pgn:
            # game is None when file is empty, or end of file is reached
            game = chess.pgn.read_game(pgn)
            while game is not None:
                boards: BughouseBoards = game.board()
                for move in game.mainline_moves():
                    for board in boards:
                        for m in board.legal_moves:
                            for c in chess.COLORS:
                                for b in chess.variant.BOARDS:
                                    mid = data.transform_chessmove_to_move_id(
                                        m, perspective_board_id=b, perspective_color=c)
                                    transformed = data.transform_move_id_to_chessmove(
                                        mid, boards, perspective_board_id=b, perspective_color=c)
                                    for k in m.__dict__:
                                        if k != "move_time":
                                            vm = m.__dict__[k]
                                            vt = transformed.__dict__[k]
                                            self.assertEqual(
                                                vm, vt, msg="Moves {} and {} differ in field {} ({} vs {})".format(
                                                    m, transformed, k, vm, vt))
                    boards.push(move)
                game = chess.pgn.read_game(pgn)