import random
from time import sleep

import chess
from chess.variant import BughouseBoards

boards = BughouseBoards()
boards.disable_pocket_saving = True

while True:
    # boards[0].push(random.choice(list(boards[0].legal_moves)))
    pocket_counts = []
    print("###################")
    print(boards)
    for i in range(100):
        b = boards[random.randint(0, 1)]
        if not any(True for _ in b.legal_moves):
            break
        move = list(b.legal_moves)[i * 2341 % len(list(b.legal_moves))]
        b.push(move)
        pocket_count = [
            [
                {p: boards[bi].pockets[c].count(p) for p in chess.PIECE_TYPES}
                for c in [False, True]]
            for bi in [0, 1]
        ]
        pocket_counts.append(pocket_count)
        print(i + 1, move)
        print(boards)

    print("undoing")
    while len(boards.move_stack) > 0:
        for bi in chess.variant.BOARDS:
            for c in chess.COLORS:
                for p in chess.PIECE_TYPES:
                    if boards[bi].pockets[c].count(p) != pocket_counts[len(boards.move_stack) - 1][bi][c][p]:
                        x = c, p
                        assert False
        print(boards)
        prev = boards.move_stack[-1]
        boards.pop()
    print(boards)
