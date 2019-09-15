from chess import Move
from engine.game import BughouseGame

game = BughouseGame()
game.start_clocks()

game[0].push(Move.from_uci("e2e4"))

print(game.clocks_s)