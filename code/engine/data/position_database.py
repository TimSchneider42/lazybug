import hashlib
import json
import logging
import math
import os.path
import shutil
import time
from typing import Optional, Generator, Sequence, Dict, Union

import numpy as np
import sys

from filelock import FileLock

from engine.data import bpgn
from engine.util import fmt_sec
from .representation import Moves


class PositionDatabase:
    def __init__(self, database_dir: str):
        self.__database_dir = database_dir
        self.__file_lock: Optional[FileLock] = None
        self.__database_open = False
        self.__metadata = None
        self.__logger = logging.getLogger(__name__)
        self.__load_metadata()

    def __enter__(self):
        self.lock()
        return self

    def __exit__(self, type, value, traceback):
        self.release()

    def lock(self):
        try:
            self.__file_lock = FileLock(os.path.join(self.__database_dir, ".lock"))
            self.__file_lock.acquire(timeout=0)
        except TimeoutError:
            raise ValueError("Database is locked.")
        try:
            self.__load_metadata()
            self.__database_open = True
        except:
            self.__file_lock.release()
            self.__file_lock = None
            raise

    def release(self):
        if self.__database_open:
            self.__database_open = False
            self.__file_lock.release()
            self.__file_lock = None

    def __store_metadata(self):
        self.__store_metadata_path(self.__database_dir, self.__metadata)

    def __load_metadata(self):
        self.__metadata = self.__load_metadata_path(self.__database_dir)

    @property
    def __metadata_filename(self) -> str:
        return self.__metadata_filename_path(self.__database_dir)

    @classmethod
    def __load_metadata_path(cls, database_dir: str) -> Dict:
        with open(cls.__metadata_filename_path(database_dir)) as f:
            return json.load(f)

    @classmethod
    def __store_metadata_path(cls, database_dir: str, metadata: Dict):
        with open(cls.__metadata_filename_path(database_dir), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def __metadata_filename_path(cls, database_dir):
        return os.path.join(database_dir, "metadata.json")

    @classmethod
    def create_empty(cls, database_dir: str, batch_size: int) -> "PositionDatabase":
        assert not os.path.exists(database_dir)
        os.mkdir(database_dir)
        metadata = {
            "num_batches": 0,
            "batch_size": batch_size,
            "sources": {}
        }
        cls.__store_metadata_path(database_dir, metadata)
        return PositionDatabase(database_dir)

    def __check_locked(self):
        assert self.__database_open

    def import_from_bpgn(self, bpgn_files: Sequence[str], batch_count: Optional[int] = None, moves_till_checkmate: Optional[int] = None):
        self.__check_locked()
        self.__logger.info("Importing positions into database...")
        remainder_positions = []
        remainder_count = 0
        remainder_path = os.path.join(self.__database_dir, "remainder")
        next_id = self.num_batches
        if os.path.exists(remainder_path):
            remainder_positions.append(Moves.load(remainder_path))
            remainder_count += len(remainder_positions[-1].move_ids)
        for bpgn_file in bpgn_files:
            if batch_count is not None and batch_count <= 0:
                break
            num_batches = 0
            skip_positions = 0
            file_completed = True
            with open(bpgn_file, "rb") as f:
                bytes = f.read()  # read entire file as bytes
                source_hash = hashlib.sha256(bytes).hexdigest()
            if source_hash in self.__metadata["sources"]:
                if self.__metadata["sources"][source_hash]["completed"]:
                    continue
                num_batches = self.__metadata["sources"][source_hash]["num_batches"]
                skip_positions = num_batches * self.batch_size
            self.__logger.info("Reading file \"{}\"...".format(bpgn_file))
            for idx, moves in enumerate(bpgn.read_batches(
                    [bpgn_file], batch_size=self.batch_size, return_incomplete=True,
                    max_positions=None if batch_count is None else batch_count * self.batch_size + 1,
                    skip_first_n_positions=skip_positions, moves_till_checkmate=moves_till_checkmate)):
                if batch_count == 0:
                    file_completed = False
                else:
                    if len(moves) == self.batch_size:
                        if batch_count is not None:
                            batch_count -= 1
                            num_batches += 1
                        self.__store_batch_at(moves, next_id)
                        self.__logger.info("Created batch with id {}".format(next_id))
                        next_id += 1
                    else:
                        remainder_positions.append(moves)
                        remainder_count += len(moves)
                        if remainder_count >= self.batch_size:
                            moves = Moves.concatenate(remainder_positions)
                            self.__store_batch_at(moves[:self.batch_size], next_id)
                            self.__logger.info("Created remainder batch with id {}".format(next_id))
                            next_id += 1
                            batch_count -= 1
            self.__logger.info("Done reading file \"{}\".".format(bpgn_file))

            self.__metadata["sources"][source_hash] = {
                "num_batches": num_batches,
                "filename": os.path.basename(bpgn_file),
                "completed": file_completed
            }
        self.__metadata["num_batches"] = next_id
        if os.path.exists(remainder_path):
            shutil.rmtree(remainder_path)
        if len(remainder_positions) > 0:
            Moves.concatenate(remainder_positions).save(remainder_path)
        self.__store_metadata()
        self.__logger.info("Done importing positions into database.")

    def __getitem__(self, item) -> Union[Moves, Generator[Moves, None, None]]:
        if isinstance(item, slice):
            return self.__iterate_batches(range(item.start, item.stop, item.step))
        elif isinstance(item, int):
            return self.__get_batch(item)
        elif isinstance(item, Sequence):
            return self.__iterate_batches(item)
        else:
            raise ValueError("Index must be integer, slice or Sequence, not {}".format(type(item)))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self.__iterate_batches(range(self.num_batches))

    def __get_batch(self, batch_id: int, log: bool = True) -> Moves:
        batch_start_time_s = time.time()

        batch_count = self.num_batches
        moves = Moves.load(os.path.join(self.__database_dir, str(batch_id)))

        batch_time_s = time.time() - batch_start_time_s
        if log:
            self.__logger.debug(
                "Read batch with id {}/{} in {}".format(batch_id, batch_count - 1, fmt_sec(batch_time_s)))

        return moves

    def __iterate_batches(self, batch_ids: Sequence[int]) -> Generator[Moves, None, None]:
        is_atty = os.isatty(sys.stdout.fileno())
        batch_start_time_s = time.time()
        total_time_s = 0

        for i, batch_id in enumerate(batch_ids):
            moves = self.__get_batch(batch_id)

            t = time.time()
            batch_time_s = t - batch_start_time_s
            total_time_s += batch_time_s
            self.__logger.debug("Read batch {}/{} with id {}/{} in {} (total time: {})".format(
                i + 1, len(batch_ids), batch_id, self.num_batches - 1, fmt_sec(batch_time_s), fmt_sec(total_time_s)))
            if is_atty:
                print("\r", end="")
                print("Read batch {}/{} with id {}/{} in {} (total time: {})".format(
                    i + 1, len(batch_ids), batch_id, self.num_batches - 1, fmt_sec(batch_time_s),
                    fmt_sec(total_time_s)), end="")

            yield moves
            batch_start_time_s = time.time()

    def __store_batch_at(self, batch: Moves, batch_id: int):
        path = os.path.join(self.__database_dir, str(batch_id))
        if os.path.exists(path):
            shutil.rmtree(path)
        batch.save(path)

    def __shuffle_batches(self, batch_ids: Sequence[int]):
        moves = list(self[batch_ids])
        cmove = Moves.concatenate(moves)
        indices = np.random.permutation(len(cmove))
        i = 0
        for bid, m in zip(batch_ids, moves):
            self.__store_batch_at(cmove[indices[i:i + len(m)]], bid)
            i += len(m)

    def shuffle(self, shuffle_with_count: int = 4, iterations: Optional[int] = None):
        self.__check_locked()
        if iterations is None:
            iterations = int(math.ceil(math.log(self.batch_size, shuffle_with_count)))
        self.__logger.info("Shuffling batches...")
        assert self.num_batches > shuffle_with_count
        batch_range = np.arange(self.num_batches)
        for i in range(iterations):
            self.__logger.info("Iteration {}/{}".format(i + 1, iterations))
            for bid in batch_range:
                other_bids = np.random.choice(batch_range[:-1] + (batch_range[:-1] >= bid), size=(shuffle_with_count,),
                                              replace=False)
                self.__shuffle_batches([bid] + other_bids.tolist())
        self.__logger.info("Done shuffling batches.")

    @property
    def batch_size(self):
        return self.__metadata["batch_size"]

    @property
    def num_batches(self):
        return self.__metadata["num_batches"]
