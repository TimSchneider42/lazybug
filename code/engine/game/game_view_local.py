from queue import Empty
from multiprocessing import Queue
from typing import Optional

import chess
from .game_view import GameView
from .local_game_messages import MessageTypes, Message


class GameViewLocal(GameView):
    def __init__(self, own_color: bool, incoming_msg_queue: Optional[Queue] = None,
                 outgoing_msg_queue: Optional[Queue] = None, clock_time_limit_s: float = 120,
                 clock_increment_s: float = 0, fen: Optional[str] = None):
        super().__init__(own_color, push_own_moves=False, clock_time_limit_s=clock_time_limit_s,
                         clock_increment_s=clock_increment_s, fen=fen)
        self.__incoming_msg_queue = Queue() if incoming_msg_queue is None else incoming_msg_queue
        self.__outgoing_msg_queue = Queue() if outgoing_msg_queue is None else outgoing_msg_queue
        self.__new_move = None

    def _send_move(self, move: chess.Move):
        self.__outgoing_msg_queue.put(Message(MessageTypes.MOVE, move))

    def updates_available(self) -> bool:
        return not self.__incoming_msg_queue.empty()

    def receive_updates(self, timeout: Optional[float] = None):
        first_update = True
        while first_update or not self.__incoming_msg_queue.empty():
            first_update = False
            try:
                self.__handle_message(self.__incoming_msg_queue.get(timeout=timeout))
            except Empty:
                pass

    def __handle_message(self, msg: Message):
        if msg.type == MessageTypes.MOVE:
            self._push(msg.data)
        elif msg.type == MessageTypes.GAME_STARTED:
            t, fen = msg.data
            self.set_fen(fen)
            self.start_clocks(t)
        elif msg.type == MessageTypes.PTELL:
            self._add_ptell_message(msg.data)
        elif msg.type == MessageTypes.GAME_FINISHED:
            self._finish_game()
        elif msg.type == MessageTypes.NAME:
            self.set_player_name(msg.data["board_id"], msg.data["color"], msg.data["name"])

    def set_own_name(self, name: str):
        self.__outgoing_msg_queue.put(Message(MessageTypes.NAME, name))

    @property
    def outgoing_msg_queue(self) -> Queue:
        return self.__outgoing_msg_queue

    @property
    def incoming_msg_queue(self) -> Queue:
        return self.__incoming_msg_queue
