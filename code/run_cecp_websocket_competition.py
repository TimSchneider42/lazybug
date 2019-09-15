#!/usr/bin/env python3
import os

from engine.game import GameViewCECPWebsocket
from engine.player import PlayerMCTS
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compete with others")
    parser.add_argument("host_address", type=str, help="Host address")
    args = parser.parse_args()
    game_view = GameViewCECPWebsocket(args.host_address)
    model_path = os.path.realpath("../models/competition_model")
    print("Using model \"{}\"".format(model_path))
    player = PlayerMCTS(model_path, max_gpu_mem_fraction=0.45, max_move_time_s=5.0, initial_random_moves=2)
    player.run_game(game_view)
