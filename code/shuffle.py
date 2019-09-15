import argparse
import logging
import os

import sys

from engine.data import PositionDatabase

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
handler.setFormatter(formatter)
LOGGER = logging.getLogger(__name__)

is_atty = os.isatty(sys.stdout.fileno())
root = logging.getLogger()
root.setLevel(logging.INFO)
root.handlers = []
root.addHandler(handler)

parser = argparse.ArgumentParser(description="Shuffles a position database.")
parser.add_argument("database_dir", type=str, help="Location of the database")
parser.add_argument("-s", "--shuffle-with-count", type=int, default=4,
                    help="Number of batches to shuffle each batch with.")
args = parser.parse_args()

with PositionDatabase(args.database_dir) as db:
    db.shuffle(args.shuffle_with_count)
