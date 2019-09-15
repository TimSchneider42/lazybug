import random
from time import sleep

import chess
from chess.variant import BughouseBoards, SingleBughouseBoard


boards = BughouseBoards()
i = 0

while not boards.is_game_over():
    i += 1
    board: SingleBughouseBoard = random.choice(boards.boards)
    if len(list(board.legal_moves)) == 0:
        board = [b for b in boards.boards if b is not board][0]

    # There can be no legal moves left without the game being over
    if len(list(board.legal_moves)) == 0:
        print("No legal moves left...")
        break
    move: chess.Move = random.choice(list(board.legal_moves))
    board.push(move)
    sleep(1.0)
    print("Move {}:".format(i))
    print(boards)
    print("")

print(boards.result())
if boards.is_threefold_repetition():
    print("Threefold repetition")
