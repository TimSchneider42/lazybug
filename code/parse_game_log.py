import os
import sys
from collections import defaultdict
from itertools import chain

path = sys.argv[1]
player_wins = defaultdict(lambda: 0)
player_draws = defaultdict(lambda: 0)
player_losses = defaultdict(lambda: 0)
for file in os.listdir(path):
    if file.endswith(".bpgn"):
        filename = os.path.join(os.path.abspath(path), file)
        with open(filename) as f:
            content = f.read()
        lines = [l.strip() for l in content.split(os.linesep)]
        team_a_names = [l.split("\"")[1] for l in lines if l.startswith("[WhiteA") or l.startswith("[BlackB")]
        team_b_names = [l.split("\"")[1] for l in lines if l.startswith("[WhiteB") or l.startswith("[BlackA")]
        result = [l.split("\"")[1] for l in lines if l.startswith("[Result ")][0]
        if result != "*" and result != "1/2-1/2":
            for n in team_a_names if result == "1-0" else team_b_names:
                player_wins[n] += 1
            for n in team_b_names if result == "1-0" else team_a_names:
                player_losses[n] += 1
        else:
            for n in team_a_names + team_b_names:
                player_draws[n] += 1

for n in set(chain(player_wins.keys(), player_losses.keys(), player_draws.keys())):
    print("{}: {}/{}/{}".format(n, player_wins[n], player_draws[n], player_losses[n]))

