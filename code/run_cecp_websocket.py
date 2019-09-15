#!/usr/bin/env python3
import os
import sys
from threading import Thread

from PyQt5 import QtWidgets

from engine.game import Visualization, GameViewCECPWebsocket
from engine.player import PlayerRandomAggressive, PlayerPolicy, PlayerMCTS


def game_runner(game_view):
    model_path = os.path.realpath("../models/current")
    print("Using model \"{}\"".format(model_path))
    if len(sys.argv) == 1 or sys.argv[1] != "policy":
        player = PlayerMCTS(model_path, max_gpu_mem_fraction=0.20, min_move_time_s=5.0, max_move_time_s=10.0,
                            initial_random_moves=0, parallel_evaluations=16)
    else:
        player = PlayerPolicy(model_path, max_gpu_mem_fraction=0.10)
    player.run_game(game_view)


if __name__ == "__main__":
    board_view = GameViewCECPWebsocket("ws://localhost:8080/websocketclient")
    app = QtWidgets.QApplication(sys.argv)
    mainWin = Visualization(board_view, refresh_views=False)
    game_thread = Thread(target=game_runner, args=(board_view,))
    game_thread.start()
    mainWin.show()
    exit_code = app.exec_()
    game_thread.join()
    sys.exit(exit_code)
