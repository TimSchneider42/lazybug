import os
import time
from collections import deque
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional, Sequence, List, Tuple
import numpy as np

import chess.variant
from engine.data import Positions
from engine.data.representation import transform_chessmove_to_move_id
from engine.game import GameView, OWN_BOARD
from engine.player.evaluator import Evaluator
from engine.player.mcts_node import TransformedMove, MCTSNode, WAIT_MOVE


class MCTS:
    def __init__(self, evaluator: Evaluator, parallel_evaluations: int = 8, logging_directory: Optional[str] = None):
        self.__base_logging_directory = logging_directory
        self.__logging_directory = None
        self.parallel_evaluations = parallel_evaluations
        self.__game_view: Optional[GameView] = None
        self.__previous_root = None
        self.__previous_move_number = 0
        self.__evaluator = evaluator
        self.partner_is_sitting = False
        self.explanation = ""

    def start_new_game(self, game_view: GameView):
        if self.__logging_directory is not None:
            self.__logging_directory = "{}_{}".format(
                self.__base_logging_directory, chess.COLOR_NAMES[game_view.own_color])
        self.__game_view = game_view
        self.__previous_root = None
        self.partner_is_sitting = False
        self.__previous_move_number = 0
        self.explanation = ""

    def _sub_selection(self, root: MCTSNode, game_view: GameView, positions: Sequence[Positions],
                       start_clocks: np.ndarray, expected_move_time_deltas: np.ndarray) \
            -> Optional[MCTSNode]:
        current_node = root
        current_clocks = start_clocks
        move = current_node.explore(np.random.choice(chess.variant.BOARDS, p=[0.9, 0.1]), current_clocks)
        if move is None:
            return None
        last_move_times = np.ones((2,)) * root.node_time_s
        while move in current_node.children and not current_node.children[move].is_virtual and move is not WAIT_MOVE:
            current_node = current_node.children[move]
            if not current_node.is_checkmate:
                move_time = current_node.node_time_s
                last_move_times[move.board_id] = move_time
                game_view.push_simulation(move, time_s=move_time)
                current_clocks = game_view.get_clocks_s(time_s=move_time)
                move = current_node.explore(np.random.choice(chess.variant.BOARDS), current_clocks)
                if move is None:
                    game_view.clear_simulated_moves()
                    return None
            else:
                game_view.clear_simulated_moves()
                return current_node
        move_time_s = None
        if move is not WAIT_MOVE:
            turn = int(game_view[move.board_id].turn)
            expected_move_time_delta = max(
                0, expected_move_time_deltas[move.board_id][turn] - last_move_times[move.board_id])
            move_time_delta = min(
                expected_move_time_delta,
                np.min(game_view.get_clocks_s(time_s=current_node.node_time_s)) / 2)
            move_time_s = current_node.node_time_s + move_time_delta
            game_view.push_simulation(move, time_s=move_time_s)
        if game_view.is_checkmate() or move is WAIT_MOVE:
            if game_view.is_checkmate():
                board_id = \
                    chess.variant.BOARD_A if game_view[chess.variant.BOARD_A].is_checkmate() else chess.variant.BOARD_B
                value = 1.0 if \
                    (game_view[board_id].turn == chess.BLACK) == (board_id == chess.variant.BOARD_A) else -1.0
                new_node = current_node.add_child(
                    move.board_id, np.array([[], []]), [self.__game_view[b].turn for b in chess.variant.BOARDS],
                    node_time_s=move_time_s, is_checkmate=True)
            else:
                clocks_s = game_view.get_clocks_s(time_s=current_node.node_time_s)
                loser_board = np.argmin([clocks_s[b][int(game_view[b].turn)] for b in chess.variant.BOARDS])
                value = -1.0 if (game_view[int(loser_board)].turn == chess.WHITE) == \
                                (int(loser_board) == chess.variant.BOARD_A) else 1.0
                new_node = current_node.add_wait_move_child(
                    [self.__game_view[b].turn for b in chess.variant.BOARDS], node_time_s=current_node.node_time_s)
            new_node.realize(np.zeros((2, 0)), value)
            game_view.clear_simulated_moves()
            return new_node

        for b in chess.variant.BOARDS:
            Positions.from_boards(game_view, current_clocks, game_view.clock_increment_s, b,
                                  game_view[b].turn, positions[b])
        legal_moves = [[
            TransformedMove(m, transform_chessmove_to_move_id(m, b, game_view[b].turn))
            for m in game_view[b].legal_moves
        ] for b in chess.variant.BOARDS]
        turns = [self.__game_view[b].turn for b in chess.variant.BOARDS]
        game_view.clear_simulated_moves()
        return current_node.add_child(move.board_id, legal_moves, turns, node_time_s=move_time_s, is_checkmate=False)

    def _selection(self, root: MCTSNode, game_view: GameView, start_clocks: np.ndarray,
                   expected_move_time_deltas: np.ndarray):
        expand_list = []
        positions = Positions.create_empty(self.parallel_evaluations * 2)
        for i in range(self.parallel_evaluations):
            sub_selection = self._sub_selection(
                root, game_view, [positions[2 * i:2 * i + 1], positions[2 * i + 1:2 * i + 2]], start_clocks,
                expected_move_time_deltas)
            if sub_selection is not None:
                expand_list.append(sub_selection)
            else:
                break
        return expand_list, positions

    def _expansion(self, virtual_nodes: Sequence[MCTSNode], positions: Positions) -> List[MCTSNode]:
        result_list = []
        positions_to_evaluate = positions[:len(virtual_nodes) * 2]
        if len(positions_to_evaluate) > 0:
            move_probs, values = self.__evaluator.evaluate_positions(positions_to_evaluate)
            for node, value, move_prob in zip(virtual_nodes, values.reshape((len(virtual_nodes), 2)),
                                              move_probs.reshape((len(virtual_nodes), 2, -1))):
                if node.is_virtual:
                    val = [
                        value[b] if (node.turns[b] == chess.WHITE) == (b == OWN_BOARD) else -value[b]
                        for b in chess.variant.BOARDS]
                    node.realize(move_prob, float(np.mean(val)))
                result_list.append(node)
        return result_list

    def _backpropagate(self, node_list: List[MCTSNode]):
        for node in node_list:
            node.backpropagate(node.value)

    def compute_next_move(self, min_time_s: float, max_time_s: float, play_random: bool = False) -> chess.Move:
        self.explanation = ""
        try:
            self.__game_view.disable_pocket_saving = True
            start_time_s = time.time()
            start_clocks = self.__game_view.clocks_s

            self.start_pockets = [[self.__game_view.boards[b].pockets[c].pieces.copy() for c in reversed(chess.COLORS)]
                                  for b in chess.variant.BOARDS]

            # Compute average move time for each player of the last 5 moves
            move_times = np.array([[
                ([self.__game_view.clock_time_limit_s + 2 * i for i in range(4, -1, -1)] +
                 [m.move_time for i, m in enumerate(self.__game_view[b].move_stack[-10:]) if (i - c) % 2 == 0])[-5:]
                for c in chess.COLORS]
                for b in chess.variant.BOARDS])

            move_timedeltas = move_times[:, :, :-1] - move_times[:, :, 1:]
            average_move_times = np.mean(move_timedeltas, axis=-1)

            root = None  # self.__previous_root
            while root is not None and self.__previous_move_number < len(
                    self.__game_view[OWN_BOARD].move_stack):
                move = self.__game_view[OWN_BOARD].move_stack[self.__previous_move_number]
                if move in root.children:
                    root = root.children[move]
                else:
                    root = None
                self.__previous_move_number += 1

            if root is None:
                positions = Positions.create_empty(2)
                turn = self.__game_view[OWN_BOARD].turn
                other_turn = self.__game_view[int(not OWN_BOARD)].turn
                Positions.from_boards(self.__game_view, start_clocks, self.__game_view.clock_increment_s, OWN_BOARD,
                                      turn, positions[0:1])
                Positions.from_boards(self.__game_view, start_clocks, self.__game_view.clock_increment_s,
                                      int(not OWN_BOARD), other_turn, positions[1:2])
                legal_moves = [
                    [TransformedMove(m, transform_chessmove_to_move_id(m, b, self.__game_view[b].turn)) for m in
                     self.__game_view[b].legal_moves] for b in chess.variant.BOARDS]
                root = MCTSNode(legal_moves, [self.__game_view[b].turn for b in chess.variant.BOARDS],
                                node_time_s=start_time_s, is_checkmate=self.__game_view.is_checkmate())
                self._expansion([root], positions)
            if play_random or max_time_s < min_time_s:
                moves = list(root.move_probabilities[OWN_BOARD].items())
                probabilities = [p for m, p in moves]
                if play_random:
                    move_index = np.random.choice(np.arange(len(probabilities)),
                                                  p=probabilities / np.sum(probabilities))
                else:
                    move_index = np.argmax(probabilities)
                return moves[move_index][0]
            max_futures = 5
            with ThreadPoolExecutor(max_workers=max_futures) as executor:
                expansion_futures = deque(maxlen=max_futures)
                n = 0
                search = True
                while search:
                    exp_list, pos = self._selection(root, self.__game_view, start_clocks,
                                                    expected_move_time_deltas=average_move_times)
                    n += len(exp_list)
                    if len(expansion_futures) == max_futures or \
                            len(expansion_futures) > 0 and expansion_futures[0].done():
                        self._backpropagate(expansion_futures.pop().result())
                    expansion_futures.append(executor.submit(self._expansion, exp_list, pos))
                    diff = time.time() - start_time_s
                    if diff >= max_time_s:
                        search = False
                    elif diff < min_time_s:
                        search = True
                    else:
                        move, visit_count = \
                            max(((move, node.games) for move, node in root.children.items()), key=lambda x: x[1])
                        # Continue search if not clear what to do
                        search = visit_count > root.games * 0.4

                while len(expansion_futures) > 0:
                    self._backpropagate(expansion_futures.pop().result())
            if self.__logging_directory is not None:
                root.create_graph(
                    os.path.join(self.__logging_directory,
                                 "{:04d}".format(self.__game_view[OWN_BOARD].fullmove_number)))
            self.__previous_root = root
            self.__previous_move_number = len(self.__game_view[OWN_BOARD].move_stack)
            if len(root.children.items()) == 0:
                moves = list(root.move_probabilities[OWN_BOARD].items())
                probabilities = [p for m, p in moves]
                idx = np.argmax(probabilities)
                return moves[idx][0]
            fact = -1 if self.__game_view.own_color == chess.BLACK else 1.0
            print("Evaluated {} nodes in {:0.2f}/{:0.2f}s - value raw: {}, acc: {}".format(
                n, time.time() - start_time_s, max_time_s, fact * root.value, fact * root.mean_value))

            own_board_visit_count = sum(c.games for m, c in root.children.items() if m.board_id == OWN_BOARD)
            other_board_visit_count = root.games - own_board_visit_count

            board_clocks = np.array([start_clocks[b][int(self.__game_view[b].turn)] for b in chess.variant.BOARDS])
            loser_board = np.argmin(board_clocks)
            own_victory_by_waiting = \
                ((self.__game_view[int(loser_board)].turn == self.__game_view[OWN_BOARD].turn)
                 != (int(loser_board) == OWN_BOARD)) and np.all(board_clocks - board_clocks[loser_board]) > 2
            partner_turn = self.__game_view[OWN_BOARD].turn != self.__game_view[int(not OWN_BOARD)].turn
            can_sit = own_victory_by_waiting or (partner_turn and not self.partner_is_sitting)
            if other_board_visit_count > 1.4 * own_board_visit_count and can_sit:
                best_other_board_move, best_other_board_visit_count = \
                    max(((move, node.games) for move, node in root.children.items() if move.board_id != OWN_BOARD),
                        key=lambda x: x[1])
                self.explanation = "best move on other board is {}, which is better than any of my moves".format(
                    best_other_board_move.uci())
                return WAIT_MOVE

            best_own_move, best_own_board_visit_count = \
                max(((move, node.games) for move, node in root.children.items() if move.board_id == OWN_BOARD),
                    key=lambda x: x[1], default=(None, None))
            if best_own_move is None:
                best_own_move = WAIT_MOVE
            if best_own_move is WAIT_MOVE:
                self.explanation = "my position is improving by just waiting"
            return best_own_move
        finally:
            self.__game_view.disable_pocket_saving = False
