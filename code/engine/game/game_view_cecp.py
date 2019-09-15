import itertools
import time
from abc import abstractmethod
from queue import Queue, Empty
from typing import List, Optional

import chess
from chess.variant import SingleBughouseBoard
from engine.game import GameView, GameAborted, OWN_BOARD
from engine.game.game_view import GameAbortReasons


class GameViewCECP(GameView):
    def __init__(self):
        super().__init__(chess.WHITE)
        self._cmd_queue = Queue()
        self.__terminating = False
        self.set_own_name("?")

    @abstractmethod
    def _send_command(self, cmd: str):
        pass

    def _add_command(self, cmd: str):
        self._cmd_queue.put((cmd, time.time()))

    def __send_cmd(self, cmd: str, *params: str):
        self._send_command(" ".join(itertools.chain([cmd], params)))

    def receive_updates(self, timeout: Optional[float] = None):
        first_update = True
        while first_update or not self._cmd_queue.empty():
            first_update = False
            try:
                raw_cmd, time_s = self._cmd_queue.get(timeout=timeout)
                cmd, *params = raw_cmd.strip().split()
                cmd = cmd.lower()
                self.handle_cmd(cmd, params, time_s)
            except Empty:
                pass

    def _reset(self):
        super()._reset()

    def handle_cmd(self, cmd: str, params: List[str], time_s: float):
        board: SingleBughouseBoard = self.boards[OWN_BOARD]
        ignored_cmds = ["xboard", "random", "computer", "accepted", "error", "partner", "otim", "holding", "pholding"]
        if cmd in ignored_cmds or cmd.startswith("#"):
            # ignore
            pass
        elif cmd == "protover":
            if params[0] != "4":
                self.__send_error("unknown protocol version {}, expected 4".format(params[0]), cmd)
            self._send_command(
                "feature san=1, time=1, variants=\"bughouse\", otherboard=1, colors=1, time=1, done=1, "
                "myname=\"{}\"".format(self.player_names[OWN_BOARD][self.own_color]))
        elif cmd == "new":
            if self.game_started:
                self._reset()
        elif cmd == "ptell":
            self._add_ptell_message(" ".join(params))
        elif cmd == "variant":
            assert params[0].lower() == "bughouse", "Cannot play variant {}".format(params[0])
        elif cmd == "quit":
            raise GameAborted(GameAbortReasons.QUIT_SIGNAL)
        elif cmd == "time":
            self.clock_time_limit_s = int(params[0]) / 100.0
        elif cmd in ["go", "playother"]:
            if cmd == "go":
                if self.own_color != board.turn:
                    self._set_own_color(board.turn)
            elif cmd == "playother":
                if self.own_color == board.turn:
                    self._set_own_color(not board.turn)
            if not self.game_started:
                self.start_clocks(time_s=time_s)
        elif cmd == "move" or cmd == "pmove":
            board_id = OWN_BOARD if cmd == "move" else int(not OWN_BOARD)
            if len(params) != 1:
                self.__send_error("illegal number of parameters - expected 1, got {}".format(len(params)), cmd)
            else:
                uci = params[0]
                move = None
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    self.__send_error("corrupt move {}".format(uci), cmd)
                if move is not None:
                    if not self.boards[board_id].is_legal(move):
                        self.__send_cmd("Illegal move:", uci)
                    else:
                        move.board_id = board_id
                        self._push(move, time_s=time_s)
                        if board_id == OWN_BOARD:
                            if not self.game_started:
                                self.start_clocks(time_s=time_s)
                            if self.own_color != board.turn:
                                self._set_own_color(board.turn)
        else:
            self.__send_error("unknown command", cmd)

    def __send_error(self, message: str, cmd: str):
        self.__send_cmd("Error", "({}):".format(message), cmd)

    def updates_available(self) -> bool:
        return not self._cmd_queue.empty()

    def _send_move(self, move: chess.Move):
        assert self.game_started
        self.__send_cmd("move", move.uci())

    def send_ptell(self, msg: str):
        self._send_command("tellics ptell {}".format(msg))

    def terminate(self):
        self.__terminating = True

    def set_own_name(self, name: str):
        assert not self.game_started
        self.set_player_name(OWN_BOARD, self.own_color, name)

    @property
    def terminating(self) -> bool:
        return self.__terminating
