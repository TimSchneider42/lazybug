#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime
import os
from typing import Generator, List

import gc
import numpy as np
import sys

from keras.optimizers import SGD

from engine.data import read_batches, Moves, PositionDatabase
from engine.data.representation import Positions
from engine.network import Network


def create_link(target: str, name: str):
    if os.path.lexists(name):
        os.unlink(name)
    os.symlink(target, name)


def data_size(data: Moves) -> int:
    return sum([f.nbytes if isinstance(f, np.ndarray) else data_size(f) for f in data])


def read_data_bpgn(paths: List[str], batch_size: int, skip_samples: int = 0) \
        -> Generator[Moves, None, None]:
    try:
        data_gen = read_batches(paths, batch_size + skip_samples)
        indices = np.arange(0, batch_size, skip_samples + 1)
        while True:
            LOGGER.info("Loading next batch...")
            gc.collect()
            output_data = {
                "prior_positions": {
                    "pieces": np.zeros((batch_size, 2, 12, 8, 8), dtype=np.bool_),
                    "pockets": np.zeros((batch_size, 2, 2, 5), dtype=np.float_),
                    "castlings": np.zeros((batch_size, 2, 2, 2), dtype=np.bool_),
                    "en_passants": np.zeros((batch_size, 2, 8, 8), dtype=np.bool_),
                    "turns": np.zeros((batch_size, 2, 2), dtype=np.bool_),
                    "remaining_times": np.zeros((batch_size, 2, 2), dtype=np.float_),
                },
                "results": np.zeros((batch_size,), dtype=np.float_),
                "move_ids": np.zeros((batch_size,), dtype=np.int16)
            }
            n = (batch_size + skip_samples) // (skip_samples + 1)
            for i in range(0, batch_size, n):
                data = next(data_gen)
                s = slice(i, i + n)
                for k, v in output_data.items():
                    d = getattr(data, k)
                    if isinstance(v, dict):
                        for k_, v_ in v.items():
                            v_[s] = getattr(d, k_)[indices][:batch_size - i]
                    else:
                        v[s] = d[indices][:batch_size - i]
                gc.collect()
            pos = Positions(**output_data["prior_positions"])
            moves = Moves(pos, output_data["move_ids"], output_data["results"])
            LOGGER.info("Done loading batch.")
            yield moves
    except StopIteration:
        pass


def read_database(database: PositionDatabase) -> Generator[Moves, None, None]:
    with database as db:
        yield from db


parser = argparse.ArgumentParser(description="Run the training.")
parser.add_argument("model_path", type=str, help="Output directory.")
data_group = parser.add_mutually_exclusive_group(required=True)
data_group.add_argument("-d", "--database-dir", type=str, default=None, help="Data to load.")
data_group.add_argument("-b", "--bpgn-data-paths", type=str, default=None, nargs="+", help="*.bpgn files to load.")
parser.add_argument("-n", "--num-batches", type=int, default=50,
                    help="Number of batches to load at once. Only relevant if --bpgn-data-paths is set.")
parser.add_argument("-g", "--gpus", type=int, default=1, help="Number of GPUs.")
parser.add_argument("-r", "--residual-blocks", type=int, default=8, help="Number of residual blocks.")
args = parser.parse_args()

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
handler.setFormatter(formatter)
LOGGER = logging.getLogger(__name__)

is_atty = os.isatty(sys.stdout.fileno())
if not is_atty:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers = []
    root.addHandler(handler)
else:
    LOGGER.addHandler(handler)

network = Network(nr_of_boards=2, num_residual_blocks=args.residual_blocks, optimizer=SGD(lr=0.001), gpus=args.gpus,
                  verbose=is_atty, value_weight=0.1, policy_weight=0.9)
network.compile_model()
print(network.model.summary())

name_prefix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
rel_model_dir = name_prefix
model_dir = os.path.join(args.model_path, rel_model_dir)
base_name = os.path.join(name_prefix + "_{:02d}_{:03d}.h5py")
os.makedirs(model_dir, exist_ok=False)
for e in range(10):
    LOGGER.info("Main epoch {}".format(e))
    if args.database_dir is not None:
        data_generator = read_database(PositionDatabase(args.database_dir))
    else:
        data_generator = read_data_bpgn(args.bpgn_data_paths, 256 * args.num_batches, 0)
    for i, b in enumerate(data_generator):
        LOGGER.info("Training on batch {} ({}MB)...".format(i, data_size(b) // 1024 ** 2))
        network.train(b, epochs=10)

        # Save model
        LOGGER.info("Saving model...")
        model_name = base_name.format(e, i)
        model_path = os.path.join(model_dir, model_name)
        network.model.save(model_path)
        create_link(model_name, os.path.join(model_dir, "latest"))
        create_link(rel_model_dir, os.path.join(args.model_path, "latest"))
        create_link(os.path.join("latest", "latest"), os.path.join(args.model_path, "latest_model"))

        LOGGER.info("Done.")
