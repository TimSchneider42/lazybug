import argparse
import logging
import os
import sys

from engine.data.position_database import PositionDatabase

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

parser = argparse.ArgumentParser(description="Load bpgn data into a database.")
parser.add_argument("data_paths", type=str, nargs="+", help="*.bpgn files to load.")
parser.add_argument("database_dir", type=str, help="Database directory directory.")
parser.add_argument("-n", "--num-batches", type=int, default=None, help="Number of batches to create.")
parser.add_argument("-b", "--batch-size", type=int, default=256 * 128,
                    help="Batch size. Will be ignored if the database already exists.")
parser.add_argument("-m", "--moves_till_checkmate", type=int, default=None,
                    help="If set only adds moves that are exactly the mth move before checkmate. \
                    Only regards games that end in checkmate. Adds exactly one move per game.")
args = parser.parse_args()

LOGGER.info("Parsing bpgn data...")
if not os.path.exists(args.database_dir):
    PositionDatabase.create_empty(args.database_dir, args.batch_size)
with PositionDatabase(args.database_dir) as db:
    db.import_from_bpgn(args.data_paths, args.num_batches, args.moves_till_checkmate)
LOGGER.info("Parsing done.")
