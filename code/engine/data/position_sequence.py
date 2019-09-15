import logging
from typing import Optional

from keras.utils import Sequence

from engine.data import PositionDatabase, Moves, Positions


class PositionSequence(Sequence):
    def __init__(self, database: PositionDatabase, batch_size: int, start: int = 0, end: Optional[int] = None,
                 rotate: int = 0):
        self.__database = database
        self.__logger = logging.getLogger(__name__)
        if self.__database.batch_size % batch_size != 0:
            self.__logger.warning("Batch size {} is not a divider of database batch size {}".format(
                batch_size, self.__database.batch_size))
        self.__cached_batch: Optional[Moves] = None
        self.__cached_batch_id: Optional[int] = None
        self.__batches_per_db_batch = self.__database.batch_size // batch_size
        if end is not None:
            assert end <= self.__batches_per_db_batch * self.__database.num_batches
            self.__batch_count = end - start
        else:
            self.__batch_count = self.__batches_per_db_batch * self.__database.num_batches - start
        self.__batch_size = batch_size
        self.__start = start
        self.__rotate = rotate

    def __len__(self) -> int:
        return self.__batch_count

    def __getitem__(self, item: int):
        assert 0 <= item < len(self)
        item = (item + self.__rotate) % len(self)
        db_batch_id, inner_id = divmod(item + self.__start, self.__batches_per_db_batch)
        if self.__cached_batch_id != db_batch_id:
            self.__logger.debug("Loading db batch {}".format(db_batch_id))
            self.__cached_batch = self.__database[db_batch_id]
            self.__cached_batch_id = db_batch_id
        index = inner_id * self.__batch_size
        moves = self.__cached_batch[index: index + self.__batch_size]
        pos: Positions = moves.prior_positions
        return [pos.en_passants, pos.pieces, pos.promotions, pos.pockets, pos.turns, pos.remaining_times, pos.castlings],\
               [moves.move_ids, moves.results]
