import argparse
import sys
from threading import Thread

from PyQt5 import QtWidgets

import chess.pgn
from engine.game import LocalGame, Visualization, PlayerSpec
from engine.player import PlayerReplay

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a *.bpgn game")
    parser.add_argument("bpgn_file", type=str, help="Filename of the game to replay.")
    args = parser.parse_args()

    with open(args.bpgn_file, encoding="iso-8859-1") as pgn:
        # game is None when file is empty, or end of file is reached
        chess_game = chess.pgn.read_game(pgn)

    time_limit, increment = [float(e) for e in chess_game.headers["TimeControl"].split("+")]

    moves = list(chess_game.mainline_moves())

    player_specs = []
    for i in range(2):
        lst_i = []
        moves_i = [m for m in moves if m.board_id == i]
        for c in range(2):
            moves_c = moves_i[int(not c):][::2]
            color = "White" if c else "Black"
            board = "B" if i else "A"
            lst_i.append(
                PlayerSpec(PlayerReplay, (moves_c,), {"name": chess_game.headers[color + board].split("\"")[0]}))
        player_specs.append(lst_i)

    player_A = player_specs[0][1]
    player_a = player_specs[0][0]
    player_B = player_specs[1][1]
    player_b = player_specs[1][0]
    game = LocalGame(player_A, player_a, player_B, player_b, time_limit_s=time_limit, time_increment_s=increment)
    obs = game.get_observer()

    # Init
    game_thread = Thread(target=game.run)
    game_thread.start()
    app = QtWidgets.QApplication(sys.argv)
    mainWin = Visualization(obs)
    mainWin.show()
    exit_code = app.exec_()
    game.abort()
    game_thread.join()

    sys.exit(exit_code)
