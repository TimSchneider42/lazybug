import sys
from threading import Thread

from PyQt5 import QtWidgets

from engine.game import LocalGame, Visualization, PlayerSpec
from engine.player import PlayerRandomAggressive, PlayerMCTS, PlayerPolicy


if __name__ == "__main__":
    path = "../../../models_by_user/rv09ivin/latest_model"
    players = [PlayerSpec(PlayerPolicy, (), {"model_path": path,"max_wait_time": 1.0}) for _ in range(4)]
    players[1] = (PlayerSpec(PlayerMCTS, (),
                                  {"max_gpu_mem_fraction": 0.25, "max_move_time_s": 5.0,"initial_random_moves": 3}))
    players[3] = (PlayerSpec(PlayerMCTS, (),
                                         {"max_gpu_mem_fraction": 0.25, "max_move_time_s": 5.0}))
    game = LocalGame(*players)
    app = QtWidgets.QApplication(sys.argv)
    mainWin = Visualization(game.get_observer())
    game_thread = Thread(target=game.run)
    game_thread.start()
    mainWin.show()
    exit_code = app.exec_()
    game.abort()
    game_thread.join()
    sys.exit(exit_code)
