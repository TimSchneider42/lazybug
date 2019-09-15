from enum import Enum
from typing import NamedTuple, Any


class MessageTypes(Enum):
    MOVE = 0
    GAME_STARTED = 1
    GAME_FINISHED = 2
    PTELL = 3
    NAME = 4


Message = NamedTuple("Message", (("type", MessageTypes), ("data", Any)))
