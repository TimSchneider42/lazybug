#!/usr/bin/env python3

import glob
import json
import os
import signal
import subprocess
import argparse
import time
import traceback
from datetime import datetime
from itertools import product
from subprocess import TimeoutExpired
from tempfile import NamedTemporaryFile
from typing import Dict

TEAMS = [["A", "b"], ["a", "B"]]

parser = argparse.ArgumentParser(description="Run a tournament between the groups.")
parser.add_argument("engine_dir", type=str, help="Engine directory.")
parser.add_argument("result_dir", type=str, help="Output directory for game results.")
parser.add_argument("mode", choices=["single", "team", "mixed"],
                    help="Whether each group should play alone or be matched with a team mate from another group.")
parser.add_argument("-t", "--time", type=int, default=120, help="Time limit per player in seconds.")
parser.add_argument("-r", "--repetitions", type=int, default=5, help="Number of games per configuration.")
args = parser.parse_args()

engine_dir = os.path.realpath(args.engine_dir)
result_dir = os.path.realpath(args.result_dir)

groups = []
# Find all groups present
for d in sorted(os.listdir(args.engine_dir)):
    if os.path.exists(os.path.join(engine_dir, d, "run.sh")):
        groups.append(d)

# Compute all configurations
pairs = []
if args.mode in ["single", "mixed"]:
    pairs += [(g, g) for g in groups]
if args.mode in ["team", "mixed"]:
    pairs += [(g1, g2) for g1, g2 in product(groups, groups) if g1 != g2]

games = [(p1, p2) for i, p1 in enumerate(pairs) for p2 in pairs[i:] if not any(g in p1 for g in p2)] * args.repetitions

# Read default config
own_path = os.path.dirname(__file__)
server_path = os.path.join(own_path, "tiny-chess-server")
config_path = os.path.join(own_path, "config.json")
with open(config_path) as f:
    default_config = json.load(f)

timeout_s = args.time * 2 + 120

# Determine tournament no.
os.makedirs(result_dir, exist_ok=True)
tournament_numbers = [int(os.path.split(t)[-1].split("_")[0]) for t in
                      glob.glob(os.path.join(result_dir, "[0-9][0-9]*_*"))]
tournament_no = max(tournament_numbers, default=0) + 1

date_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

tournament_dir = os.path.join(result_dir, "{:03d}_{}".format(tournament_no, date_string))

os.mkdir(tournament_dir)

metadata = {
    "tournament_no": tournament_no,
    "participants": groups,
    "wins": {
        g: 0 for g in groups
    },
    "losses": {
        g: 0 for g in groups
    },
    "draws": {
        g: 0 for g in groups
    },
    "games": {}
}

print("Tournament no. {}".format(tournament_no))

# Run the games
for game_numer, game_configuration in enumerate(games):
    game_number = game_numer + 1

    game_metadata = {}
    metadata["games"][game_number] = game_metadata

    bpgn_filename = "{:03d}_{:03d}_{}.bpgn".format(tournament_no, game_number, date_string)
    bpgn_path = os.path.join(tournament_dir, bpgn_filename)

    game_metadata["configuration"] = {
        pn: pp for tn, tp in zip(TEAMS, game_configuration) for pn, pp in zip(tn, tp)
    }

    gc = game_configuration
    print("Game {}/{}: \"{}\" and \"{}\" vs \"{}\" and \"{}\"".format(
        game_number, len(games), gc[0][0], gc[0][1], gc[1][0], gc[1][1]))

    # Create server config
    config = {}
    config.update(default_config)

    # Write config
    config["game"]["time"] = args.time
    config["application"]["save"]["save_dir"] = tournament_dir
    config["application"]["save"]["filename"] = bpgn_filename
    config["application"]["save"]["meta"]["round"] = tournament_no
    config["application"]["save"]["meta"]["game_no"] = game_number

    result = "*"
    result_comment = ""

    try:
        with NamedTemporaryFile(mode="w", delete=False) as config_file:
            json.dump(config, config_file)
            config_filename = config_file.name

        # Start server
        print("Starting server process...")
        server_process = subprocess.Popen(["node", "index.js", config_filename], cwd=server_path)

        # Give server time to start
        time.sleep(3.0)

        # Start clients
        client_processes: Dict[str, subprocess.Popen] = {}
        proc_group_ids = []
        for t, (tn, tp) in enumerate(zip(TEAMS, game_configuration)):
            for p, (pn, pp) in enumerate(zip(tn, tp)):
                print("Starting player \"{}\"...".format(pn))
                connection_string = "ws://localhost:{}/websocketclient?{}=''".format(
                    config["httpServer"]["port"], pn)
                group_dir = os.path.join(engine_dir, game_configuration[t][p])
                env = dict(os.environ)
                env.update({
                    "CUDA_VISIBLE_DEVICES": str(t),
                    "HOME": group_dir
                })
                proc = subprocess.Popen(
                    ["/bin/bash", os.path.join(group_dir, "run.sh"), connection_string, str(tournament_no),
                     str(game_number), pn], stdout=open(os.devnull, 'w'), stderr=subprocess.PIPE, cwd=group_dir,
                    preexec_fn=os.setsid, env=env)
                try:
                    proc_group_ids.append(os.getpgid(proc.pid))
                except ProcessLookupError:
                    print("Warning: could not find process of player {} right after starting it".format(pn))
                client_processes[game_configuration[t][p]] = proc

        try:
            print("Waiting for server to terminate...")
            server_process.wait(timeout=timeout_s)
        except TimeoutExpired:
            print("Timed out waiting for server. Sending SIGINT...")
            server_process.send_signal(signal.SIGINT)
            for p in client_processes.values():
                p.send_signal(signal.SIGINT)
            try:
                server_process.wait(timeout=10)
            except TimeoutExpired:
                print("Server not terminating. Killing it...")
                server_process.send_signal(signal.SIGKILL)

        print("Server terminated.")

        wait_start_time = time.time()

        engines_alive = None

        print("Waiting for all engines to terminate...")
        while engines_alive is None or len(engines_alive) > 0 and time.time() - wait_start_time < 15:
            engines_alive = [n for n, p in client_processes.items() if p.poll() is None]

            if not len(engines_alive) == 0:
                print("Waiting for engines \"{}\"".format("\", \"".join(engines_alive)))
                time.sleep(1.0)

        if len(engines_alive) > 0:
            print("Killing engines \"{}\" as they failed to terminate.".format("\", \"".join(engines_alive)))

            for pp in engines_alive:
                client_processes[pp].kill()

        # Kill whatever is left of the engines
        for pg in proc_group_ids:
            try:
                os.killpg(pg, signal.SIGKILL)
            except ProcessLookupError:
                pass

        # Determine winner
        with open(bpgn_path) as f:
            for l in f.readlines():
                if l.startswith("[ResultComment"):
                    result_comment = l.strip()[1:-1].split(maxsplit=1)[-1][1:-1]
                elif l.startswith("[Result"):
                    result = l.strip()[1:-1].split(maxsplit=1)[-1][1:-1]
    except Exception as e:
        result_comment = "Game crashed with exception \"{}\"".format(e)
        traceback.print_exc()
    finally:
        try:
            os.remove(config_file)
        except:
            pass

    game_metadata["result"] = result
    game_metadata["result_comment"] = result_comment
    if result == "1/2-1/2":
        for p in metadata["draws"]:
            metadata["draws"][p] += 1
    elif result in ["1-0", "0-1"]:
        for p in range(2):
            metadata["wins"][gc[0 if result == "1-0" else 1][p]] += 1
            metadata["losses"][gc[1 if result == "1-0" else 0][p]] += 1

    print("Game complete.")

with open(os.path.join(tournament_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=True)

print("Tournament complete.")
