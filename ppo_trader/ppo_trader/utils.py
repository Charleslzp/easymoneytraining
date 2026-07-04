"""常量定义"""
from enum import IntEnum


class Actions(IntEnum):
    OPEN_LONG = 0
    OPEN_SHORT = 1
    CLOSE = 2
    HOLD = 3
    DO_NOTHING = 4


class States(IntEnum):
    EMPTY = 0
    LONG = 1
    SHORT = 2
