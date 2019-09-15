from time import sleep
from typing import Sequence

import chess
from engine.game import OWN_BOARD
from engine.player import Player


class PlayerReplay(Player):
    def __init__(self, moves: Sequence[chess.Move], name: str = "LazyBug - Replay"):
        self.__moves = moves
        self.__move_index = 0
        super().__init__(name)

    def move(self):
        if self.__move_index < len(self.__moves):
            move = self.__moves[self.__move_index]
            my_clock = self.game_view.clocks_s[OWN_BOARD][int(self.game_view.own_color)]
            wait_time = my_clock - move.move_time
            if wait_time > 0:
                sleep(wait_time)
            self.__move_index += 1
            move.board_id = OWN_BOARD
            while not self.game_view.is_legal(move):
                self.game_view.receive_updates()
            return move
        else:
            sleep(0.1)
            return None

    def on_game_started(self):
        self.__move_index = 0