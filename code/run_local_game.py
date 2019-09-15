import os
import argparse
import sys
from threading import Thread

from PyQt5 import QtWidgets

from chess.variant import BughouseBoards
from engine.game import LocalGame, Visualization, PlayerSpec
from engine.player import PlayerRandomAggressive, PlayerMCTS, PlayerPolicy, PlayerHuman

MODEL_PATH = "../models/current"


_terminate = False


def _boards_changed_event_handler(sender, args):
    print(obs)


def _obs_updater():
    while not _terminate and not obs.is_game_over():
        obs.receive_updates(timeout=0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a local game.')
    parser.add_argument("-v", "--visualize", action="store_true", default=False)
    args = parser.parse_args()

    logging_base_dir = "../logs"
    os.makedirs(logging_base_dir, exist_ok=True)
    logging_dir = os.path.join(logging_base_dir, "{:04d}".format(len(os.listdir(logging_base_dir))))
    # logging_dir = None

    player_spec_random = PlayerSpec(PlayerRandomAggressive, (), {"max_wait_time": 15.0, "min_wait_time": 5.0})
    player_spec_random_fast = PlayerSpec(PlayerRandomAggressive, (), {"max_wait_time": 2.0, "min_wait_time": 1.0})
    player_spec_mcts = PlayerSpec(PlayerMCTS, (MODEL_PATH,),
                                  {"max_gpu_mem_fraction": 0.15, "max_move_time_s": 10.0, "min_move_time_s": 5.0,
                                   "logging_directory": logging_dir, "initial_random_moves": 3,
                                   "parallel_evaluations": 16})
    player_spec_mcts_simple = PlayerSpec(PlayerMCTS, (),
                                         {"max_gpu_mem_fraction": 0.25, "max_move_time_s": 5.0})
    player_spec_policy_fast = PlayerSpec(PlayerPolicy, (MODEL_PATH,),
                                         {"max_wait_time": 2.0, "min_wait_time": 1.0, "max_gpu_mem_fraction": 0.10,
                                          "verbose": False})
    player_spec_policy = PlayerSpec(PlayerPolicy, (MODEL_PATH,),
                                    {"max_wait_time": 7.0, "min_wait_time": 3.0, "max_gpu_mem_fraction": 0.10})
    newstdin = os.fdopen(os.dup(sys.stdin.fileno()))
    player_spec_human = PlayerSpec(PlayerHuman, (newstdin,), {})

    print("Using model \"{}\"".format(os.path.realpath(MODEL_PATH)))
    player_A = player_spec_mcts
    player_b = player_spec_mcts
    player_B = player_spec_policy_fast
    player_a = player_spec_policy_fast
    game = LocalGame(player_A, player_a, player_B, player_b, time_limit_s=300)
    obs = game.get_observer()

    fen = BughouseBoards.starting_fen
    # fen = "rn1qkbnr/pp3ppp/8/1p1bq3/3n1K2/3P1PP1/PPP2p1P/R1BQ2NR[pnbPPNBQ] w KQkq - 0 1 | " \
    #       "4rk2/p1p2prp/2N2p2/3p4/3P1Bb1/2P5/P1P2PPP/R3KBNR[pp] w KQkq - 1 1"

    # Init
    game_thread = Thread(target=game.run, args=(fen,))
    game_thread.start()
    obs_thread = None
    exit_code = None
    if args.visualize:
        app = QtWidgets.QApplication(sys.argv)
        mainWin = Visualization(obs)
        mainWin.show()
        exit_code = app.exec_()
        game.abort()
    else:
        obs.game_state_changed_event.add(_boards_changed_event_handler)
        obs_thread = Thread(target=_obs_updater)
        obs_thread.start()
        obs_thread.join()
    game_thread.join()

    if exit_code is not None:
        sys.exit(exit_code)
