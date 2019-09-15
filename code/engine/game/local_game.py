import time
from queue import Empty
from multiprocessing import Process, Queue, Event
from typing import Optional, Tuple, List, NamedTuple, Type, Any, Dict

import chess.variant
import chess
from .local_game_messages import MessageTypes, Message
from .bughouse_game import BughouseGame, PLAYER_NAMES
from .game_view_local import GameViewLocal
from .game_view import OWN_BOARD, OTHER_BOARD

PlayerSpec = NamedTuple("PlayerSpec",
                        (("player_type", Type["Player"]), ("args", Tuple[Any, ...]), ("kwargs", Dict[str, Any])))

_PlayerData = NamedTuple("_PlayerData", (
    ("player_spec", PlayerSpec), ("input_queue", Queue), ("output_queue", Queue), ("ready_event", Event)))


class LocalGame:
    def __init__(self, player_A: PlayerSpec, player_a: PlayerSpec, player_B: PlayerSpec, player_b: PlayerSpec,
                 time_limit_s: float = 120.0, time_increment_s: float = 0.0):
        self.__time_limits_s = time_limit_s
        self.__time_increments_s = time_increment_s
        self.__player_data: Optional[Tuple[Tuple[_PlayerData, ...], ...]] = None
        self.__player_specs = ((player_a, player_A), (player_b, player_B))
        # Each side has an own copy of the boards object to ensure thread safety
        self.__observers: List[GameViewLocal] = []

        self.__main_game: Optional[BughouseGame] = None
        self.__abort = False

    def _create_player(self, spec: PlayerSpec) -> _PlayerData:
        return _PlayerData(spec, Queue(), Queue(), Event())

    def run(self, starting_fen: Optional[str] = None):
        assert self.__main_game is None, "The game has already been started."
        self.__player_data = tuple(tuple(self._create_player(s) for s in bs) for bs in self.__player_specs)
        self.__main_game = BughouseGame(clock_time_limit_s=self.__time_limits_s,
                                        clock_increment_s=self.__time_increments_s, fen=starting_fen)
        self.__abort = False

        player_processes = [
            Process(target=LocalGame._player_proc, args=(p, bool(i), self.__time_limits_s, self.__time_increments_s))
            for board_id in range(len(self.boards))
            for i, p in enumerate(self.__player_data[board_id])
        ]
        for proc in player_processes:
            proc.start()

        player_ready = [[False, False], [False, False]]

        print("Waiting for players...")
        while not all(all(pr for pr in br) for br in player_ready):
            for bi, bpd in enumerate(self.__player_data):
                for pi, pd in enumerate(bpd):
                    if not player_ready[bi][pi]:
                        if pd.ready_event.wait(0.01):
                            print("Player {} ready.".format(PLAYER_NAMES[bi][pi]))
                            player_ready[bi][pi] = True

        now = time.time()
        for bi, bpd in enumerate(self.__player_data):
            for pd in bpd:
                if bi == chess.variant.BOARD_A:
                    fen = self.__main_game.fen()
                else:
                    fen_split = self.__main_game.fen().split("|")
                    fen = "|".join(reversed(fen_split))
                pd.input_queue.put(Message(MessageTypes.GAME_STARTED, (now, fen)))

        for b in self.__observers:
            b.incoming_msg_queue.put(Message(MessageTypes.GAME_STARTED, (now, self.__main_game.fen())))
        self.__main_game.start_clocks(now)
        print("All players ready. Game started.")

        while not self.__main_game.is_game_over() and not self.__abort:
            for board_id in range(len(self.boards)):
                other_board_id = int(not bool(board_id))
                player_data_own = self.__player_data[board_id]
                player_data_other = self.__player_data[other_board_id]
                board = self.__main_game[board_id]

                for color in chess.COLORS:
                    try:
                        msg = player_data_own[int(color)].output_queue.get(timeout=0.01)
                    except Empty:
                        continue

                    now = time.time()
                    if msg is not None:
                        if msg.type == MessageTypes.MOVE:
                            move: chess.Move = msg.data
                            if not self.__main_game.is_game_over(time_s=now):
                                move.board_id = OWN_BOARD
                                move.move_time = self.__main_game.get_clock_s(board_id, now)[int(board.turn)]
                                move_other_perspective = move.__copy__()
                                move_other_perspective.board_id = OTHER_BOARD
                                move_main = move if board_id == OWN_BOARD else move_other_perspective
                                assert color == board.turn, \
                                    "Player {} ({}) attempted to play move {} but it was not his turn.".format(
                                        PLAYER_NAMES[board_id][color],
                                        self.__main_game.player_names[board_id][color], move_main)
                                assert self.__main_game.is_legal(
                                    move_main), "Player {} ({}) attempted to play illegal move {}".format(
                                    PLAYER_NAMES[board_id][color], self.__main_game.player_names[board_id][color],
                                    move_main)
                                self.__main_game.push(move_main)
                                for b in self.__observers:
                                    b.incoming_msg_queue.put(Message(MessageTypes.MOVE, move_main))
                                for p in player_data_own:
                                    p.input_queue.put(Message(MessageTypes.MOVE, move))
                                for p in player_data_other:
                                    p.input_queue.put(Message(MessageTypes.MOVE, move_other_perspective))
                        elif msg.type == MessageTypes.PTELL:
                            print("{} ({}) -> {} ({}): \"{}\"".format(
                                PLAYER_NAMES[board_id][color], self.__main_game.player_names[board_id][color],
                                PLAYER_NAMES[not board_id][not color],
                                self.__main_game.player_names[board_id][not color], msg))
                            player_data_other[int(not color)].input_queue.put(msg)
                        elif msg.type == MessageTypes.NAME:
                            self.__main_game.set_player_name(board_id, color, msg.data)
                            for bid, bpd in enumerate(self.__player_data):
                                msg_data = {
                                    "name": msg.data,
                                    "board_id": 0 if board_id == bid else 1,
                                    "color": color
                                }
                                for pd in bpd:
                                    pd.input_queue.put(Message(MessageTypes.NAME, msg_data))
                            for b in self.__observers:
                                msg_data = {
                                    "name": msg.data,
                                    "board_id": board_id,
                                    "color": color
                                }
                                b.incoming_msg_queue.put(Message(MessageTypes.NAME, msg_data))

        for b in self.__observers:
            b.incoming_msg_queue.put(Message(MessageTypes.GAME_FINISHED, None))
        for bpd in self.__player_data:
            for pd in bpd:
                pd.input_queue.put(Message(MessageTypes.GAME_FINISHED, None))

        print("Result: {} (Player {})".format(self.__main_game.result(), self.__main_game.result_comment()))
        print("Waiting for processes to terminate...")

        for p in player_processes:
            p.join()

        print("All processes terminated.")

    def abort(self):
        self.__abort = True

    @staticmethod
    def _player_proc(player_data: _PlayerData, color: bool, clock_time_limit_s: float, clock_increment_s: float):
        spec = player_data.player_spec
        player = spec.player_type(*spec.args, **spec.kwargs)
        player_data.ready_event.set()
        player.run_game(GameViewLocal(color, player_data.input_queue, player_data.output_queue,
                                      clock_time_limit_s=clock_time_limit_s, clock_increment_s=clock_increment_s))

    def get_observer(self):
        obs = GameViewLocal(chess.WHITE, clock_time_limit_s=self.__time_limits_s,
                            clock_increment_s=self.__time_increments_s)
        self.__observers.append(obs)
        return obs

    @property
    def boards(self) -> Optional[chess.variant.BughouseBoards]:
        return self.__main_game

    @property
    def done(self) -> bool:
        return self.boards.is_game_over()

    @property
    def player_specs(self):
        return self.__player_specs
