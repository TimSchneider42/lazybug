#!/usr/bin/env python3
import os
from datetime import datetime
import logging
logging.basicConfig(filename="{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), level=logging.DEBUG)

from engine.game import GameViewCECPStdin
from engine.player import PlayerRandomAggressive, PlayerPolicy

if __name__ == "__main__":
    player = PlayerPolicy(os.path.realpath("../models/latest_model"))
    game_view = GameViewCECPStdin()
    player.run_game(game_view)