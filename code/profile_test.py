import time
from queue import Queue

from chess.variant import BughouseBoards
from engine.game import GameViewLocal, OWN_BOARD
from engine.game.local_game_messages import Message, MessageTypes
from engine.player.evaluator_network import EvaluatorNetwork
from engine.player.mcts import MCTS
from engine.player.mcts_node import WAIT_MOVE

q = Queue()
view = GameViewLocal(True, q, q)
q.put(Message(MessageTypes.GAME_STARTED, (time.time(), BughouseBoards.starting_fen)))
view.receive_updates()
mcts = MCTS(EvaluatorNetwork("../models/current"), parallel_evaluations=32, logging_directory="../profile")
mcts.start_new_game(view)
# mcts = MCTSSimple(200, 0.0, 0, 0.5)

while not view.is_game_over() and not view.fullmove_number() > 100:
    moves = list(view[0].legal_moves)
    view._set_own_color(view[OWN_BOARD].turn)
    if len(moves) != 0:
        m = mcts.compute_next_move(5.0, 5.0)
        if m is not WAIT_MOVE:
            view.push(m)
    print(view)