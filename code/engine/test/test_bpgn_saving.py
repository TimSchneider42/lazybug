import datetime

import chess
import chess.pgn
import chess.variant

boards = chess.variant.BughouseBoards()

move = chess.Move.from_uci("e2e4")
move.board_id = 0
move.move_time = 115.3  # Clock of the player moving

boards.push(move)

move = chess.Move.from_uci("d2d4")
move.board_id = 1
move.move_time = 113.2

boards.push(move)

move = chess.Move.from_uci("e7d5")
move.board_id = 0
move.move_time = 110.9

boards.push(move)

game = chess.pgn.Game.from_bughouse_boards(boards)
game.headers["WhiteA"] = "kevin"
game.headers["WhiteB"] = "juffi"
game.headers["BlackA"] = "john"
game.headers["BlackB"] = "alfred"
game.headers["TimeControl"] = "120+0"
game.headers["Event"] = "TU Darmstadt Bughouse Championship"
game.headers["Site"] = "?"
game.headers["Date"] = datetime.datetime.now().isoformat()
game.headers["Round"] = "1"

PLAYER_NAMES = [["a", "A"], ["b", "B"]]

if boards.is_threefold_repetition():
    result = "1/2-1/2"
    result_comment = "Game drawn by threefold repetition"
elif boards.is_checkmate():
    result = boards.result()
    for i, b in enumerate(boards):
        if b.is_checkmate():
            losing_player = b.turn
            result_comment = "{} checkmated".format(PLAYER_NAMES[i][losing_player])
elif False:
    result = "dummy"
    result_comment = "Player X forfeits on time"
else:
    result = "*"
    result_comment = "Game aborted"


game.headers["Result"] = result
game.headers["ResultComment"] = result_comment

with open("output.bpgn", "w", encoding="utf-8") as f:
    exporter = chess.pgn.FileExporter(f, headers=True, comments=False, variations=False)
    game.accept(exporter)
