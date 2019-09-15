import logging
from typing import Dict, Optional, Sequence, NamedTuple, Tuple
import numpy as np
from graphviz import Digraph

import chess
import chess.variant

TransformedMove = NamedTuple("TransformedMove", (("move", chess.Move), ("move_id", int)))

WAIT_MOVE = chess.Move(None, None)


class MCTSNode:
    def __init__(self, legal_moves: Sequence[Sequence[TransformedMove]], turns: Sequence[bool], node_time_s: float,
                 is_checkmate: bool, parent: Optional["MCTSNode"] = None, cpuct_base: float = 19652.0,
                 cpuct_init: float = 2.0, cpuct_factor: float = 1.5):
        self.__parent = parent
        self.__children: Dict[chess.Move, "MCTSNode"] = {}
        self.__virtual_descendant_count = 1

        # Value of team white on board A
        self.__turns = turns
        self.__value = None
        self.__games = 0
        self.__move_probabilities = None
        self.__unexplored_moves = None
        self.__accumulated_value = 0
        self.__legal_moves = legal_moves
        self.__cpuct_base = cpuct_base
        self.__cpuct_init = cpuct_init
        self.__cpuct_factor = cpuct_factor
        self.__node_time_s = node_time_s
        self.__wait_for_opponent_probability = 0.01
        self.__is_checkmate = is_checkmate

    def realize(self, move_probabilities: Sequence[np.ndarray], value: float):
        # Efficient sorting of the move probabilities
        assert self.is_virtual
        # Probabilities of legal moves for each board
        legal_move_probabilities = [
            mp[[mid for m, mid in lm]]
            for mp, lm in zip(move_probabilities, self.__legal_moves)]
        legal_mass = [np.sum(lmp) + self.__wait_for_opponent_probability for lmp in legal_move_probabilities]
        sort_indices = [np.argsort(lmp) for lmp in legal_move_probabilities]
        self.__move_probabilities = [
            {m: mp[mid] / lmass for m, mid in lm}
            for mp, lm, lmass in zip(move_probabilities, self.__legal_moves, legal_mass)]
        self.__unexplored_moves = [
            [lm[i][0] for i in lmi]
            for lm, lmi in zip(self.__legal_moves, sort_indices)]
        self.__value = value

    @property
    def is_virtual(self):
        return self.__value is None

    @property
    def turns(self) -> Sequence[bool]:
        return self.__turns

    @property
    def value(self) -> Optional[float]:
        return self.__value

    @property
    def games(self) -> int:
        return self.__games

    @property
    def mean_value(self):
        return self.__accumulated_value / self.__games

    @property
    def children(self) -> Dict[chess.Move, "MCTSNode"]:
        return self.__children

    @property
    def is_terminal(self):
        return all(len(lm) == 0 for lm in self.__legal_moves)

    @property
    def is_checkmate(self) -> bool:
        return self.__is_checkmate

    @property
    def move_probabilities(self) -> Sequence[Dict[chess.Move, float]]:
        return self.__move_probabilities

    @property
    def virtual_descendant_count(self) -> int:
        return self.__virtual_descendant_count

    @property
    def node_time_s(self) -> float:
        return self.__node_time_s

    def get_score(self, move: chess.Move) -> float:
        """
        Returns the score of the move from the perspective of the moving player
        :param move:
        :return:
        """
        # This is duplicate code for performance reasons
        if move in self.__children:
            child = self.__children[move]
            child_games = child.__games + child.__virtual_descendant_count
            if (move.board_id == chess.variant.BOARD_A) == (self.__turns[move.board_id] == chess.WHITE):
                val = child.__accumulated_value
            else:
                val = -child.__accumulated_value
            exploitation_score = (val - child.__virtual_descendant_count) / child_games
        else:
            child_games = 0
            exploitation_score = -1
        exploration_rate = self.__cpuct_factor * np.log((1 + self.games + self.__cpuct_base) / self.__cpuct_base) \
                           + self.__cpuct_init
        exploration_score = exploration_rate * self.__move_probabilities[move.board_id][move] / (child_games + 1) * \
                            np.sqrt(self.games + self.__virtual_descendant_count)
        return exploitation_score + exploration_score

    def __compute_wait_move_score(self, own_board_best_move: chess.Move, other_board_best_move: chess.Move) -> float:
        if other_board_best_move in self.__children:
            child = self.__children[other_board_best_move]
            wait_move_child_games = child.__games + child.__virtual_descendant_count
            if self.__turns[own_board_best_move.board_id] != self.__turns[other_board_best_move.board_id]:
                # The other moving player is on my team
                val = child.__accumulated_value
            else:
                val = -child.__accumulated_value
            wait_move_exploitation_score = (val - child.__virtual_descendant_count) / wait_move_child_games
        else:
            wait_move_child_games = 0
            wait_move_exploitation_score = -1

        wait_move_exploration_rate = self.__cpuct_factor * np.log(
            (1 + wait_move_child_games + self.__cpuct_base) / self.__cpuct_base) + self.__cpuct_init
        wait_move_exploration_score = \
            wait_move_exploration_rate * self.__wait_for_opponent_probability / (wait_move_child_games + 1) * \
            np.sqrt(self.games + self.__virtual_descendant_count)
        return wait_move_exploration_score + wait_move_exploitation_score

    @property
    def parent(self) -> "MCTSNode":
        return self.__parent

    def __find_best_move(self, board_id: int) -> Tuple[chess.Move, float]:
        best_move = max((m for m, n in self.__children.items() if m.board_id == board_id and not n.is_virtual),
                        key=lambda move: self.get_score(move), default=None)
        if best_move is not None:
            best_score = self.get_score(best_move)
        else:
            best_score = None

        if len(self.__unexplored_moves[board_id]) > 0:
            unexplored_move_score = self.get_score(self.__unexplored_moves[board_id][-1])
            if best_move is None or best_score < unexplored_move_score:
                best_move = self.__unexplored_moves[board_id][-1]
                best_score = unexplored_move_score
        return best_move, best_score

    def explore(self, board_id: int, clocks_s: np.ndarray) -> Optional[chess.Move]:
        own_board_id = board_id
        other_board_id = int(not own_board_id)

        own_board_best_move, own_best_score = self.__find_best_move(board_id)
        other_board_best_move, other_best_score = self.__find_best_move(other_board_id)

        board_clocks = np.array([clocks_s[b][int(self.__turns[b])] for b in chess.variant.BOARDS])
        loser_board = np.argmin(board_clocks)
        own_victory_by_waiting = \
            ((self.__turns[int(loser_board)] == self.__turns[int(own_board_id)])
             != (int(loser_board) == own_board_id)) and np.all(board_clocks - board_clocks[loser_board]) > 2
        other_victory_by_waiting = \
            ((self.__turns[int(loser_board)] == self.__turns[int(other_board_id)])
             != (int(loser_board) == other_board_id)) and np.all(board_clocks - board_clocks[loser_board]) > 2

        if own_board_best_move is None and other_board_best_move is None:
            # Both players cannot move
            return WAIT_MOVE
        elif own_board_best_move is None \
                or (self.__compute_wait_move_score(own_board_best_move, other_board_best_move) > own_best_score
                    and (own_victory_by_waiting
                         or self.__turns[int(own_board_id)] != self.__turns[int(other_board_id)])):
            # If the player cannot move or he knows that he will take advantage out of just waiting, he waits for the
            # other player to move
            if other_board_best_move is None \
                    or (self.__compute_wait_move_score(other_board_best_move, own_board_best_move) > other_best_score
                        and other_victory_by_waiting):
                # If the player on the other board cannot move or knows that he will take advantage out of just waiting,
                # he waits as well (e.g. when own board is drawn and the clock is lower on own board, then opponent on
                # other board will wait)
                return WAIT_MOVE
            return other_board_best_move
        return own_board_best_move

    def print(self, indent: int = 0, depth: Optional[int] = None):
        for move, child in sorted(self.__children.items(), key=lambda x: x[1].games):
            output = "{}: {}".format(move, child.games)
            indent_str = " " * indent * 2
            print(indent_str + output)
            if depth is None or depth > 0:
                child.print(indent + 1, None if depth is None else depth - 1)

    def create_graph(self, output_filename: str):
        try:
            graph = Digraph()
            self.__add_subtree(graph)
            graph.render(output_filename)
        except Exception as e:
            logging.getLogger(__name__).warning("Graphviz rendering failed with exception: \"{}\"".format(e))

    def __add_subtree(self, graph: Digraph, score: Optional[float] = None):
        if score is not None:
            own_label = "v: {:0.3f}\ns: {:0.3f}\ng: {}".format(self.value, score, self.games)
        else:
            own_label = "v: {:0.3f}\ng: {}".format(self.value, self.games)
        graph.node(str(self), label=own_label)
        for m, c in self.children.items():
            if c.games > 20:
                c.__add_subtree(graph, score=self.get_score(m))
                move_label = "{}\np:{:0.3f}".format(m, self.move_probabilities[m.board_id][m])
                graph.edge(str(self), str(c), label=move_label)

    def add_child(self, board_id: int, legal_moves: Sequence[Sequence[TransformedMove]], turns: Sequence[bool],
                  node_time_s: float, is_checkmate: bool) -> "MCTSNode":
        move = self.__unexplored_moves[board_id][-1]
        del self.__unexplored_moves[board_id][-1]
        return self._add_child(move, legal_moves, turns, node_time_s, is_checkmate)

    def add_wait_move_child(self, turns: Sequence[bool], node_time_s: float) -> "MCTSNode":
        return self._add_child(WAIT_MOVE, [[], []], turns, node_time_s, False)

    def _add_child(self, move: chess.Move, legal_moves: Sequence[Sequence[TransformedMove]], turns: Sequence[bool],
                   node_time_s: float, is_checkmate: bool) -> "MCTSNode":
        node = MCTSNode(legal_moves, turns, node_time_s, is_checkmate, parent=self)
        self.__children[move] = node
        current_node = self
        while current_node is not None:
            current_node.__virtual_descendant_count += 1
            current_node = current_node.__parent
        return node

    def _backpropagate(self, value: float, decrement_virtual_descendant_count: bool):
        if decrement_virtual_descendant_count:
            self.__virtual_descendant_count -= 1
            assert self.__virtual_descendant_count >= 0
        self.__accumulated_value += value
        self.__games += 1
        if self.__parent is not None:
            self.__parent._backpropagate(value, decrement_virtual_descendant_count)

    def backpropagate(self, value: float):
        self._backpropagate(value, self.__games == 0)
