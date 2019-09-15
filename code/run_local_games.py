import argparse
import os
import sys
from threading import Thread

from chess.variant import BughouseBoards
from engine.game import LocalGame, PlayerSpec
from engine.player import PlayerRandomAggressive, PlayerMCTS, PlayerPolicy, PlayerHuman

MODEL_PATH = "../models/current.h5py"
RESULT_PATH = "../results"
mcts_win_count = policy_win_count = policy_checkmated_count = mcts_checkmated_count = draw_count = 0

_terminate = False


def _boards_changed_event_handler(sender, args):
    print(obs)


def _obs_updater():
    while not _terminate and not obs.is_game_over():
        obs.receive_updates(timeout=0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a local game.')
    parser.add_argument("-v", "--visualize", action="store_true", default=False)
    parser.add_argument("-r", "--runcount", type=int, default=1, help="Number of games to run.")
    args = parser.parse_args()

    logging_base_dir = "../logs"
    os.makedirs(logging_base_dir, exist_ok=True)
    os.makedirs(RESULT_PATH, exist_ok=True)
    logging_dir = os.path.join(logging_base_dir, "{:04d}".format(len(os.listdir(logging_base_dir))))
    logging_dir = None

    player_spec_random = PlayerSpec(PlayerRandomAggressive, (), {"max_wait_time": 15.0, "min_wait_time": 5.0})
    player_spec_random_fast = PlayerSpec(PlayerRandomAggressive, (), {"max_wait_time": 2.0, "min_wait_time": 1.0})
    player_spec_mcts = PlayerSpec(PlayerMCTS, (MODEL_PATH,),
                                  {"max_gpu_mem_fraction": 0.20, "max_move_time_s": 5.0,
                                   "logging_directory": logging_dir, "initial_random_moves": 3,
                                   "parallel_evaluations": 16})
    player_spec_mcts_simple = PlayerSpec(PlayerMCTS, (),
                                         {"max_gpu_mem_fraction": 0.20, "max_move_time_s": 5.0})
    player_spec_policy_fast = PlayerSpec(PlayerPolicy, (MODEL_PATH,),
                                         {"max_wait_time": 2.0, "min_wait_time": 1.0, "max_gpu_mem_fraction": 0.10,
                                          "verbose": False})
    player_spec_policy = PlayerSpec(PlayerPolicy, (MODEL_PATH,),
                                    {"max_wait_time": 7.0, "min_wait_time": 3.0, "max_gpu_mem_fraction": 0.10})
    newstdin = os.fdopen(os.dup(sys.stdin.fileno()))
    player_spec_human = PlayerSpec(PlayerHuman, (newstdin,), {})

    print("Using model \"{}\"".format(os.path.realpath(MODEL_PATH)))
    for x in range(0, args.runcount):
        player_A = player_spec_mcts
        player_b = player_spec_mcts
        player_B = player_spec_policy_fast
        player_a = player_spec_policy_fast
        game = LocalGame(player_A, player_a, player_B, player_b, time_limit_s=120)
        obs = game.get_observer()

        fen = BughouseBoards.starting_fen
        # fen = "7k/5ppp/8/8/8/8/8/R6K[] w KQkq - 0 1 | " \
        #       "7k/8/8/8/8/8/5PPP/r6K[] w KQkq - 1 1"

        # Init
        game_thread = Thread(target=game.run, args=(fen,))
        game_thread.start()
        obs_thread = None
        exit_code = None
        obs.game_state_changed_event.add(_boards_changed_event_handler)
        obs_thread = Thread(target=_obs_updater)
        obs_thread.start()
        obs_thread.join()
        game_thread.join()
        result = game.boards.result()
        print(result)
        if result == "0-1":
            policy_win_count += 1
        elif result == "1-0":
            mcts_win_count += 1
        else:
            draw_count += 1
        result_comment = game.boards.result_comment()
        if result_comment == "a (LazyBug-Policy) checkmated" or result_comment == "B (LazyBug-Policy) checkmated":
            policy_checkmated_count += 1
        else:
            mcts_checkmated_count += 1
        f = open(RESULT_PATH + "/gamesresults.txt", "w")
        f.write(
            "Wins Policy: {} / checkmates: {} \nWins MCTS: {}/ checkmates: {} \nDraws: {} \nTotal Games: {}".format(
                policy_win_count, mcts_checkmated_count, mcts_win_count, policy_checkmated_count, draw_count, x+1
            ))
        f.close()
    sys.exit(0)
