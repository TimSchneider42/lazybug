import time
from abc import abstractmethod
from typing import Optional

import chess.variant
import chess
from engine.game import GameView, OWN_BOARD, GameAborted
from engine.game.game_view import GameAbortReasons


class Player:
    def __init__(self, name: str):
        self.__game_view: Optional[chess.variant.BughouseBoards] = None
        self.__name = name

    def run_game(self, game_view: "GameView"):
        game_view.set_own_name(self.__name)
        restart = True
        self.__game_view = game_view
        game_view.own_color_changed_event.add(self._color_switched_handler)
        while restart:
            # restart = False
            while not game_view.game_started or game_view.game_finished(time_s=time.time()):
                try:
                    game_view.receive_updates()
                except GameAborted:
                    pass
            self.on_game_started()
            try:
                while not game_view.game_finished():
                    own_turn = game_view[OWN_BOARD].turn == game_view.own_color
                    if own_turn and any(True for _ in self.game_view[OWN_BOARD].legal_moves):
                        move = self.move()
                        now = time.time()
                        if move is None:
                            game_view.receive_updates(timeout=0)
                        elif not game_view.game_finished(time_s=now):
                            move.move_time = None
                            game_view.push(move, time_s=now)
                    else:
                        game_view.receive_updates()
            except GameAborted as e:
                if e.reason == GameAbortReasons.BOARD_RESET:
                    restart = True
            self.on_game_terminated()

    def _color_switched_handler(self, sender, earg):
        self.on_color_switched()

    def on_game_started(self):
        pass

    def on_game_terminated(self):
        pass

    def on_color_switched(self):
        pass

    @abstractmethod
    def move(self) -> chess.Move:
        pass

    @property
    def game_view(self) -> Optional[GameView]:
        return self.__game_view

    @property
    def name(self) -> str:
        return self.__name

    def __repr__(self):
        return self.__name
