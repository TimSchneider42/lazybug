import os
from enum import Enum
from itertools import chain
from typing import NamedTuple, Optional, List, Sequence

import chess.pgn
import chess.variant
import numpy as np


class Winner(Enum):
    STALEMATE = 0
    THREEFOLD_REP = 1
    DRAW = 2
    CHECKMATED_A = 3
    CHECKMATED_a = 4
    CHECKMATED_B = 5
    CHECKMATED_b = 6
    TIME_A = 7
    TIME_a = 8
    TIME_B = 9
    TIME_b = 10


class WinnerSimple(Enum):
    TEAM_A = -1
    DRAW = 0
    TEAM_B = 1


_REVERSE_PIECE_DICT = dict(enumerate(chess.PIECE_TYPES))

_KNIGHT_MOVE_DICT = {
    (-2, 1): 56,
    (-1, 2): 57,
    (1, 2): 58,
    (2, 1): 59,
    (2, -1): 60,
    (1, -2): 61,
    (-1, -2): 62,
    (-2, -1): 63
}

_PIECE_DICT = {v: k for k, v in _REVERSE_PIECE_DICT.items()}

REL_MOVE_IDS = 87
NUM_MOVE_IDS = 2 * 8 * 8 * REL_MOVE_IDS


def preserve_attr_access(cls):
    for i, n in enumerate(cls._fields):
        def access(self, j=i):
            return super(cls, self).__getitem__(j)

        setattr(cls, n, property(access))
    return cls


PositionsFields = NamedTuple("PositionsFields",
                             (("pieces", np.ndarray),
                              ("pockets", np.ndarray),
                              ("turns", np.ndarray),
                              ("castlings", np.ndarray),
                              ("en_passants", np.ndarray),
                              ("promotions", np.ndarray),
                              ("remaining_times", np.ndarray),
                              ("time_increments", np.ndarray)))


@preserve_attr_access
class Positions(PositionsFields):
    BB_NP = np.array(chess.BB_SQUARES, dtype=np.uint64).reshape((8, 8))

    def __len__(self):
        return self.pieces.shape[0]

    def __getitem__(self, item) -> "Positions":
        return Positions(*(f[item] for f in super().__iter__()))

    def __setitem__(self, key, value: "Positions"):
        for i, v in enumerate(super(Positions, value)):
            super().__getitem__(i)[key] = v

    def save(self, path: str):
        os.mkdir(path)
        for pos_name in self._fields:
            val = getattr(self, pos_name)
            if pos_name == "pieces":
                val = val.astype(np.int8)
                val = np.sum([val[:, :, i] * (i + 1) for i in range(12)], axis=0)
            np.save(os.path.join(path, pos_name), val)

    @property
    def nbytes(self):
        return sum(f.nbytes for f in super().__iter__())

    @staticmethod
    def create_empty(size: int = 0) -> "Positions":
        pieces = np.zeros((size, 2, 12, 8, 8), dtype=np.bool_)
        pockets = np.zeros((size, 2, 2, 5), dtype=np.int8)
        turns = np.zeros((size, 2, 2), dtype=np.bool_)
        castlings = np.zeros((size, 2, 2, 2), dtype=np.bool_)
        en_passants = np.zeros((size, 2, 8, 8), dtype=np.bool_)
        promotions = np.zeros((size, 2, 8, 8), dtype=np.bool_)
        remaining_times = np.zeros((size, 2, 2), dtype=np.float_)
        time_increments = np.zeros((size,), dtype=np.float_)
        return Positions(pieces, pockets, turns, castlings, en_passants, promotions, remaining_times, time_increments)

    @staticmethod
    def concatenate(positions: Sequence["Positions"]) -> "Positions":
        return Positions(*(np.concatenate(
            [super(Positions, p).__getitem__(i) for p in positions], axis=0) for i in range(len(Positions._fields))))

    @staticmethod
    def from_boards(boards: chess.variant.BughouseBoards, remaining_times_s: np.ndarray, time_increment_s: float,
                    perspective_board_id: int = 0, perspective_color: bool = chess.WHITE,
                    out_positions: Optional["Positions"] = None) -> "Positions":
        """
        transforms an board to an position, which can is the input for the neural net, should only been called from
        transform_game_to_position, except when no game node is available
        :param boards: board which will be transformed to position
        :param mirror_boards: to roll or not to roll?
        :return:
        """
        if out_positions is None:
            out_positions = Positions.create_empty(1)

        board_id_order = [int(perspective_board_id), int(not perspective_board_id)]
        color_order = [int(perspective_color), int(not perspective_color)]
        out_positions.remaining_times[0] = remaining_times_s / 60.0
        out_positions.time_increments[0] = time_increment_s / 60.0
        for i, bid in enumerate(board_id_order):
            board = boards[bid]
            mirror = (perspective_color == chess.BLACK) == (bid == perspective_board_id)
            out_positions.turns[0][i][int(not (board.turn == perspective_color) == (bid == perspective_board_id))] = 1
            if board.ep_square is not None:
                out_positions.en_passants[0][i].reshape((64,))[board.ep_square] = 1
                if mirror:
                    out_positions.en_passants[0, i] = np.flipud(out_positions.en_passants[0, i])
            out_positions.promotions[0, i] = np.bitwise_and(board.promoted, Positions.BB_NP)
            if mirror:
                out_positions.promotions[0, i] = np.flipud(out_positions.promotions[0, i])
            if bid != perspective_board_id:
                color_order = reversed(color_order)
            for j, color in enumerate(color_order):
                out_positions.remaining_times[0, i, j] = remaining_times_s[bid, color] / 60.0
                out_positions.castlings[0, i, j, 0] = board.has_kingside_castling_rights(color)
                out_positions.castlings[0, i, j, 1] = board.has_queenside_castling_rights(color)
                for piece in chess.PIECE_TYPES:
                    if piece != chess.KING:
                        out_positions.pockets[0, i, j, piece - 1] = board.pockets[color].pieces[piece]
                    out_positions.pieces[0, i, j * 6 + (piece - 1)] = \
                        np.bitwise_and(board.pieces_mask(piece, color), Positions.BB_NP)
                    if mirror:
                        # Mirror board
                        out_positions.pieces[0, i, j * 6 + (piece - 1)] = np.flipud(
                            out_positions.pieces[0, i, j * 6 + (piece - 1)])
        return out_positions

    @staticmethod
    def load(file_name: str) -> "Positions":
        positions_dict = {
            pos_name: np.load(os.path.join(file_name, pos_name + ".npy")) for pos_name in Positions._fields
        }
        p = positions_dict["pieces"]
        p = np.array([p == i + 1 for i in range(12)], dtype=np.bool_).transpose((1, 2, 0, 3, 4))
        positions_dict["pieces"] = p
        return Positions(**positions_dict)


MovesFields = NamedTuple("MovesFields",
                         (("prior_positions", Positions),
                          ("move_ids", np.ndarray),
                          ("results", np.ndarray),
                          ("move_timedelta", np.ndarray),
                          ("game_id", np.ndarray),
                          ("perspective_board_id", np.ndarray),
                          ("perspective_color", np.ndarray),
                          ("move_no", np.ndarray),
                          ("player_elos", np.ndarray)))


@preserve_attr_access
class Moves(MovesFields):
    def __len__(self):
        return len(self.move_ids)

    def __getitem__(self, item) -> "Moves":
        return Moves(*(f[item] for f in super().__iter__()))

    def __setitem__(self, key, value: "Moves"):
        for i, v in enumerate(super(Moves, value).__iter__()):
            super().__getitem__(i)[key] = v

    def save(self, path: str):
        """
        saves the contents of move at folder_name
        :param move: Moves struct to save
        :param folder_name: target folder
        :return:
        """
        os.mkdir(path)
        for nm in Moves._fields:
            o = getattr(self, nm)
            directory = os.path.join(path, nm)
            if isinstance(o, np.ndarray):
                np.save(directory, o)
            else:
                o.save(directory)

    @property
    def nbytes(self):
        return sum(f.nbytes for f in super().__iter__())

    @staticmethod
    def create_empty(size: int = 0) -> "Moves":
        prior_positions = Positions.create_empty(size)
        move_ids = np.zeros((size,), dtype=np.int16)
        results = np.zeros((size,), dtype=np.float_)
        move_timedelta = np.zeros((size,), dtype=np.float_)
        game_id = np.zeros((size,), dtype=np.int_)
        perspective_board_id = np.zeros((size,), dtype=np.bool_)
        perspective_color = np.zeros((size,), dtype=np.bool_)
        move_no = np.zeros((size,), dtype=np.int_)
        player_elos = np.zeros((size, 2, 2), dtype=np.uint16)
        return Moves(prior_positions, move_ids, results, move_timedelta, game_id, perspective_board_id,
                     perspective_color, move_no, player_elos)

    @staticmethod
    def concatenate(moves: List["Moves"]) -> "Moves":
        prior_positions = Positions.concatenate([m.prior_positions for m in moves])
        other_fields = [
            np.concatenate([super(Moves, m).__getitem__(i) for m in moves], axis=0)
            for i in range(1, len(Moves._fields))
        ]
        return Moves(prior_positions, *other_fields)

    @staticmethod
    def from_boards(boards: chess.variant.BughouseBoards, result: WinnerSimple, move: chess.Move,
                    remaining_times_s: np.ndarray, time_increment_s: float, move_timedelta_s: float,
                    game_id: int, player_elos: np.ndarray, out_moves: Optional["Moves"] = None) -> "Moves":
        if out_moves is None:
            out_moves = Moves.create_empty(1)
        perspective_board_id = move.board_id
        perspective_color = boards[perspective_board_id].turn
        out_moves.perspective_color[0] = perspective_color
        out_moves.perspective_board_id[0] = perspective_board_id
        out_moves.move_no[0] = len(boards[move.board_id].move_stack)
        out_moves.game_id[0] = game_id

        board_id_order = [int(perspective_board_id), int(not perspective_board_id)]
        color_order = [int(perspective_color), int(not perspective_color)]
        for i, bid in enumerate(board_id_order):
            for j, color in enumerate(color_order if bid == perspective_board_id else reversed(color_order)):
                out_moves.player_elos[0, i, j] = player_elos[bid, color]

        Positions.from_boards(
            boards, remaining_times_s=remaining_times_s, time_increment_s=time_increment_s,
            perspective_board_id=perspective_board_id, perspective_color=perspective_color,
            out_positions=out_moves.prior_positions)
        perspective_is_team_a = (perspective_board_id == chess.variant.BOARD_A) == (perspective_color == chess.WHITE)
        if result == WinnerSimple.TEAM_A:
            result_val = 1.0 if perspective_is_team_a else -1.0
        elif result == WinnerSimple.TEAM_B:
            result_val = -1.0 if perspective_is_team_a else 1.0
        else:
            result_val = 0.0
        out_moves.results[0] = result_val
        out_moves.move_ids[0] = transform_chessmove_to_move_id(
            move, perspective_color=perspective_color, perspective_board_id=perspective_board_id)
        out_moves.move_timedelta[0] = move_timedelta_s / 60
        return out_moves

    @staticmethod
    def load(filename: str) -> "Moves":
        positions = Positions.load(os.path.join(filename, "prior_positions"))
        moves_dict = {
            moves_name: np.load(os.path.join(filename, moves_name + ".npy"))
            for moves_name in Moves._fields if moves_name != "prior_positions"
        }
        moves_dict["prior_positions"] = positions
        return Moves(**moves_dict)


def _compute_moveid(from_square: chess.Square, to_square: chess.Square, board_id: int, promotion: chess.PieceType,
                    drop: chess.PieceType, perspective_color: chess.Color, perspective_board_id: int) -> Optional[int]:
    from_square = np.array([from_square % 8, from_square // 8])
    to_square = np.array([to_square % 8, to_square // 8])
    if (perspective_color == chess.BLACK) == (board_id == perspective_board_id):
        # Roll
        from_square[1] = 7 - from_square[1]
        to_square[1] = 7 - to_square[1]

    if drop is None:
        file, rank = from_square
        move_dir = to_square - from_square
        if promotion is not None and promotion != chess.QUEEN:
            promotion_offset = _PIECE_DICT[promotion] - 1
            y_direction_offset = 0 if move_dir[1] == 1 else 9
            if move_dir[0] < 0:
                x_direction_offset = 0
            elif move_dir[0] == 0:
                x_direction_offset = 3
            else:
                x_direction_offset = 6
            rel_move_id = 64 + promotion_offset + y_direction_offset + x_direction_offset
        elif move_dir[0] == 0:
            rel_move_id = (move_dir[1] - 1) if move_dir[1] > 0 else -(move_dir[1] + 1) + 28
        elif move_dir[0] == move_dir[1]:
            rel_move_id = ((move_dir[0] - 1) if move_dir[0] > 0 else -(move_dir[0] + 1) + 28) + 7
        elif move_dir[1] == 0:
            rel_move_id = ((move_dir[0] - 1) if move_dir[0] > 0 else -(move_dir[0] + 1) + 28) + 14
        elif move_dir[0] == -move_dir[1]:
            rel_move_id = ((move_dir[0] - 1) if move_dir[0] > 0 else -(move_dir[0] + 1) + 28) + 21
        elif tuple(move_dir) in _KNIGHT_MOVE_DICT:
            rel_move_id = _KNIGHT_MOVE_DICT[tuple(move_dir)]
        else:
            return None
    else:
        file, rank = to_square
        rel_move_id = 82 + _PIECE_DICT[drop]

    if perspective_board_id != board_id:
        file += 8

    return (rel_move_id << 7) | (rank << 4) | file


MOVE_IDS = {
    (fs, ts, b, p, d, pc, pb): _compute_moveid(fs, ts, b, p, d, pc, pb)
    for fs in chess.SQUARES
    for ts in chess.SQUARES
    for b in chess.variant.BOARDS
    for p in chain(chess.PIECE_TYPES, [None])
    for d in chain(chess.PIECE_TYPES, [None])
    for pc in chess.COLORS
    for pb in chess.variant.BOARDS
}


def transform_chessmove_to_move_id(move: chess.Move, perspective_board_id: int = 0,
                                   perspective_color: bool = chess.WHITE) -> int:
    return MOVE_IDS[(move.from_square, move.to_square, move.board_id, move.promotion, move.drop, perspective_color,
                     perspective_board_id)]


def transform_move_id_to_chessmove(move_id: int, boards: chess.variant.BughouseBoards,
                                   perspective_board_id: int = 0,
                                   perspective_color: bool = chess.WHITE) -> Optional[chess.Move]:
    assert 0 <= move_id < NUM_MOVE_IDS
    rel_move_id = move_id >> 7
    move_dir = np.zeros(2, dtype=np.int_)
    promotion = None
    drop = None
    from_square = np.array([move_id & 0xF, (move_id >> 4) & 0x7])
    if from_square[0] < 8:
        board_id = int(perspective_board_id)
    else:
        board_id = int(not perspective_board_id)
        from_square[0] -= 8
    board = boards[board_id]

    assert 0 <= rel_move_id <= 86
    if rel_move_id < 56:
        if rel_move_id < 7:
            move_dir[0] = 0
            move_dir[1] = rel_move_id + 1
        elif rel_move_id < 14:
            move_dir[0] = move_dir[1] = (rel_move_id + 1) - 7
        elif rel_move_id < 21:
            move_dir[1] = 0
            move_dir[0] = (rel_move_id + 1) - 14
        elif rel_move_id < 28:
            move_dir[0] = (rel_move_id + 1) - 21
            move_dir[1] = - move_dir[0]
        elif rel_move_id < 35:
            move_dir[0] = 0
            move_dir[1] = - (rel_move_id + 1 - 28)
        elif rel_move_id < 42:
            move_dir[0] = move_dir[1] = - (rel_move_id + 1 - 28) + 7
        elif rel_move_id < 49:
            move_dir[1] = 0
            move_dir[0] = - (rel_move_id + 1 - 28) + 14
        elif rel_move_id < 56:
            move_dir[0] = - (rel_move_id + 1 - 28) + 21
            move_dir[1] = - move_dir[0]
        to_square = move_dir + from_square
    elif rel_move_id < 64:
        # knight moves
        reverse_kmd = dict([reversed(i) for i in _KNIGHT_MOVE_DICT.items()])
        move_dir = np.asarray(reverse_kmd[rel_move_id])
        to_square = move_dir + from_square
    elif rel_move_id < 82:
        # underpromotions
        if rel_move_id < 67:
            promotion = _REVERSE_PIECE_DICT[rel_move_id - 64 + 1]
            move_dir[0] = -1
            move_dir[1] = 1
        elif rel_move_id < 70:
            promotion = _REVERSE_PIECE_DICT[rel_move_id - 67 + 1]
            move_dir[0] = 0
            move_dir[1] = 1
        elif rel_move_id < 73:
            promotion = _REVERSE_PIECE_DICT[rel_move_id - 70 + 1]
            move_dir[0] = 1
            move_dir[1] = 1
        elif rel_move_id < 76:
            promotion = _REVERSE_PIECE_DICT[rel_move_id - 73 + 1]
            move_dir[0] = -1
            move_dir[1] = -1
        elif rel_move_id < 79:
            promotion = _REVERSE_PIECE_DICT[rel_move_id - 76 + 1]
            move_dir[0] = -0
            move_dir[1] = -1
        else:
            promotion = _REVERSE_PIECE_DICT[rel_move_id - 79 + 1]
            move_dir[0] = 1
            move_dir[1] = -1
        to_square = move_dir + from_square
    else:
        # drops
        drop = _REVERSE_PIECE_DICT[rel_move_id - 82]
        to_square = np.copy(from_square)

    if (perspective_color == chess.BLACK) == (board_id == perspective_board_id):
        from_square[1] = 7 - from_square[1]
        to_square[1] = 7 - to_square[1]

    assert (0 <= to_square).all() and (0 <= from_square).all() and (to_square <= 7).all() and (from_square <= 7).all()
    if rel_move_id < 56 and (to_square[1] == 7 or to_square[1] == 0) and \
            board.piece_at(chess.square(*from_square)).piece_type == chess.PAWN:
        promotion = chess.QUEEN
    return chess.Move(chess.square(*from_square), chess.square(*to_square), promotion, drop, board_id)
