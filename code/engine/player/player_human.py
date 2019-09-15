import os
from multiprocessing import Queue

import chess
from engine.game import OWN_BOARD
from engine.player import Player


class PlayerHuman(Player):
    def __init__(self, input: os.fdopen):
        self.__input = input
        super().__init__("LazyBug-Human")

    def move(self):
        move = None
        while (move is None or not self.game_view[OWN_BOARD].is_legal(move)) and \
                not self.game_view[OWN_BOARD].is_game_over():
            uci = ""
            try:
                print("Next move: ", end="")
                uci = self.__input.readline().strip()
                self.game_view.receive_updates(timeout=0.0)
                if uci != "":
                    move = chess.Move.from_uci(uci)
                    if not self.game_view[OWN_BOARD].is_legal(move):
                        print("Illegal move: {}".format(uci))
            except:
                print("Corrupt move: {}".format(uci))
        return move
