import logging
import os
import sys
import time
from typing import Optional, Generator, Sequence, List

import numpy as np

import chess.pgn
import chess.variant
from chess.pgn import LOGGER as PGN_LOGGER
from engine.util import fmt_sec
from .representation import Moves, WinnerSimple


def read_files(file_list: Sequence[str], max_positions: Optional[int] = None,
               moves_till_checkmate: Optional[int] = None) -> Moves:
    return next(
        read_batches(file_list, batch_size=None, return_incomplete=True, max_positions=max_positions,
                     moves_till_checkmate=moves_till_checkmate))


def _grab_batch(moves_list: List[Moves], batch_size: int) -> Moves:
    add_list = []
    add_list_move_count = 0
    while add_list_move_count < batch_size and len(moves_list) > 0:
        m = moves_list[0][:batch_size - add_list_move_count]
        if len(m) == len(moves_list[0]):
            moves_list.pop(0)
        else:
            moves_list[0] = moves_list[0][batch_size - add_list_move_count:]
        add_list_move_count += len(m)
        add_list.append(m)
    return Moves.concatenate(add_list)


def read_batches(file_list: Sequence[str], batch_size: Optional[int] = None, return_incomplete: bool = False,
                 max_positions: Optional[int] = None, skip_first_n_positions: int = 0,
                 moves_till_checkmate: Optional[int] = None) \
        -> Generator[Moves, None, None]:
    LOGGER = logging.getLogger(__name__)
    batch_start_time_s = time.time()
    total_time_s = 0
    moves_to_add = []
    moves_to_add_size = 0
    total_invalid_positions = 0
    total_valid_positions = 0
    batch_invalid_positions = 0
    total_processed_positions = 0
    batch_skipped_positions = 0
    moves_to_skip = skip_first_n_positions
    batch_no = 1
    last_log = time.time()
    valid_games = 0
    invalid_games = 0

    is_atty = os.isatty(sys.stdout.fileno())

    # Suppress error messages for unreadable games
    PGN_LOGGER.setLevel(100)
    for path in file_list:
        with open(path, encoding="iso-8859-1") as pgn:
            # game is None when file is empty, or end of file is reached
            game = chess.pgn.read_game(pgn)
            while game is not None and (max_positions is None or total_processed_positions < max_positions):
                boards = game.board()
                mainline_moves = list(game.mainline_moves())
                result_code = game.headers['Result']
                valid = False
                if len(mainline_moves) > 10 and len(game.errors) == 0 and (
                        moves_till_checkmate is None or len(mainline_moves) >= moves_till_checkmate):
                    if len(mainline_moves) <= moves_to_skip or (
                            moves_till_checkmate is not None and 1 <= moves_to_skip):
                        valid = True
                    else:
                        try:
                            if "TimeControl" in game.headers and result_code != "*":
                                if result_code == '1-0':
                                    result = WinnerSimple.TEAM_A
                                elif result_code == '0-1':
                                    result = WinnerSimple.TEAM_B
                                else:
                                    result = WinnerSimple.DRAW
                                game_time_s, time_increment_s = list(map(float, game.headers["TimeControl"].split("+")))
                                clocks = np.ones((2, 2)) * game_time_s
                                game_moves = Moves.create_empty(len(mainline_moves))
                                for i, move in enumerate(mainline_moves):
                                    move_timedelta_s = \
                                        clocks[move.board_id, int(boards[move.board_id].turn)] - move.move_time
                                    clocks[move.board_id, boards[move.board_id].turn] = move.move_time
                                    player_elos = np.zeros((2, 2), dtype=np.uint16)
                                    for j, b in enumerate(["A", "B"]):
                                        for k, c in enumerate(["White", "Black"]):
                                            tag = "{}{}Elo".format(b, c)
                                            if tag in game.headers:
                                                player_elos[j, k] = game.headers[tag]
                                    Moves.from_boards(boards, result=result, move=move, out_moves=game_moves[i:i + 1],
                                                      remaining_times_s=clocks, time_increment_s=time_increment_s,
                                                      move_timedelta_s=move_timedelta_s, player_elos=player_elos,
                                                      game_id=game.headers["BughouseDBGameNo"])
                                    boards.push(move)
                                valid = True
                        except:
                            LOGGER.warning("Encountered error not caught by bpgn parsing when replaying game",
                                           exc_info=sys.exc_info())

                if valid:
                    if moves_till_checkmate is not None:
                        if boards.is_checkmate:
                            valid_games += 1
                            total_valid_positions += 1
                            if moves_to_skip == 0:
                                m = game_moves[
                                    len(game_moves) - moves_till_checkmate: len(game_moves) - moves_till_checkmate + 1]
                                if max_positions is not None:
                                    m = m[:max_positions - total_processed_positions]
                                total_processed_positions += len(m)
                                moves_to_add.append(m)
                                moves_to_add_size += len(m)
                            else:
                                batch_skipped_positions += 1
                                moves_to_skip -= 1
                    else:
                        valid_games += 1
                        total_valid_positions += len(mainline_moves)
                        if moves_to_skip < len(mainline_moves):
                            m = game_moves[moves_to_skip:]
                            batch_skipped_positions += len(mainline_moves) - len(m)
                            moves_to_skip -= len(mainline_moves) - len(m)
                            if max_positions is not None:
                                m = m[:max_positions - total_processed_positions]
                            total_processed_positions += len(m)
                            moves_to_add.append(m)
                            moves_to_add_size += len(m)
                        else:
                            batch_skipped_positions += len(mainline_moves)
                            moves_to_skip -= len(mainline_moves)

                    t = time.time()
                    batch_time_s = t - batch_start_time_s
                    if is_atty:
                        print("\r", end="")
                        print("Batch {}: read valid/invalid: {}/{}, skipped {} in {}".format(
                            batch_no, moves_to_add_size, batch_invalid_positions,
                            batch_skipped_positions, fmt_sec(batch_time_s)), end="")
                    if time.time() - last_log >= 300:
                        LOGGER.info("Batch {}: read valid/invalid: {}/{}, skipped {} in {}".format(
                            batch_no, moves_to_add_size, batch_invalid_positions,
                            batch_skipped_positions, fmt_sec(batch_time_s)))
                        last_log = time.time()

                    while batch_size is not None and moves_to_add_size >= batch_size:
                        moves = _grab_batch(moves_to_add, batch_size)
                        # Pause timer
                        t = time.time()
                        batch_time_s = t - batch_start_time_s
                        total_time_s += batch_time_s
                        if is_atty:
                            print("\r", end="")
                            print("Batch {} complete.".format(batch_no))
                            print("Read {} valid, {} invalid and skipped {} positions in {}".format(
                                moves_to_add_size, batch_invalid_positions, batch_skipped_positions,
                                fmt_sec(batch_time_s)))
                            print("In total I read {} valid games ({} positions), "
                                  "{} invalid games ({} positions) and skipped {} positions in {}".format(
                                valid_games, total_valid_positions, invalid_games, total_invalid_positions,
                                skip_first_n_positions - moves_to_skip,
                                fmt_sec(total_time_s)))
                        LOGGER.info("Batch {} complete.".format(batch_no))
                        LOGGER.info("Read {} valid, {} invalid and skipped {} positions in {}".format(
                            moves_to_add_size, batch_invalid_positions, batch_skipped_positions,
                            fmt_sec(batch_time_s)))
                        LOGGER.info(
                            "In total I read {} valid games ({} positions), "
                            "{} invalid games ({} positions) and skipped {} positions in {}".format(
                                valid_games, total_valid_positions, invalid_games,
                                total_invalid_positions, skip_first_n_positions - moves_to_skip, fmt_sec(total_time_s)))
                        batch_invalid_positions = 0
                        batch_skipped_positions = 0
                        batch_no += 1
                        moves_to_add_size -= batch_size
                        yield moves

                        # Resume timer
                        batch_start_time_s = time.time()
                        last_log = time.time()
                else:
                    invalid_games += 1
                    total_invalid_positions += len(mainline_moves)
                    batch_invalid_positions += len(mainline_moves)
                game = chess.pgn.read_game(pgn)

    # If no batch size is given, we return everything in the end
    moves = None
    if moves_to_add_size > 0:
        moves = _grab_batch(moves_to_add, batch_size)

    total_time_s += (time.time() - batch_start_time_s)
    if is_atty:
        print("\r", end="")
        print("Dataset complete.")
        print("Read {} valid games ({} positions), {} invalid games ({} positions) "
              "and skipped {} positions in {}".format(
            valid_games, total_valid_positions, invalid_games, total_invalid_positions,
            skip_first_n_positions - moves_to_skip, fmt_sec(total_time_s)))
    LOGGER.info("Dataset complete.")
    LOGGER.info("Read {} valid games ({} positions), {} invalid games ({} positions) "
                "and skipped {} positions in {}".format(
        valid_games, total_valid_positions, invalid_games, total_invalid_positions,
        skip_first_n_positions - moves_to_skip, fmt_sec(total_time_s)))

    if moves is not None and (return_incomplete or batch_size is None):
        yield moves
