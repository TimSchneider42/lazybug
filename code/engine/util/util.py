from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from engine.data import Positions


def fmt_sec(seconds: float):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{}:{:02d}:{:04.1f}h".format(int(hours), int(minutes), seconds)


def position_hr(position: "Positions") -> np.ndarray:
    output_array = np.zeros((8, 16), dtype=int)
    pieces_concat = np.concatenate(position.pieces, axis=-1)
    for i in range(pieces_concat.shape[0]):
        mult = - i - 1 if i < 6 else i - 5
        output_array += pieces_concat[i] * mult
    return -output_array
