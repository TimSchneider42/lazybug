import time
import unittest

import chess.pgn
import chess.variant
import numpy as np
from engine import data


def try_convert(m, dboard, b, c):
    try:
        return data.representation.transform_move_id_to_chessmove(m, dboard, b, c)
    except:
        return None


COUNT = 16

class TestMoveTransform(unittest.TestCase):
    def test_move_id(self):
        for c in chess.COLORS:
            for b in chess.variant.BOARDS:
                dboard = chess.variant.BughouseBoards()
                moveids = np.random.choice(range(data.representation.NUM_MOVE_IDS), size=COUNT, replace=False)
                chessmoves = [try_convert(m, dboard, b, c) for m in moveids]
                moveids = [m for m, c in zip(moveids, chessmoves) if c is not None]
                chessmoves = [c for c in chessmoves if c is not None]

                start = time.time()
                for moveid, cm in zip(moveids, chessmoves):
                    returned_moveid = data.representation.transform_chessmove_to_move_id(cm, b, c)
                    self.assertEqual(returned_moveid, moveid,
                                     msg="Moveid {} was falsely converted to {} (single)".format(
                                         moveid, returned_moveid))
                diff_single = time.time() - start
                print("single: {}".format(diff_single))

                start = time.time()
                for moveid, cm in zip(moveids, chessmoves):
                    returned_moveid = data.representation.transform_chessmove_to_move_id_quick(cm, b, c)
                    self.assertEqual(returned_moveid, moveid,
                                     msg="Moveid {} was falsely converted to {} (quick)".format(
                                         moveid, returned_moveid))
                diff_single = time.time() - start
                print("quick: {}".format(diff_single))
