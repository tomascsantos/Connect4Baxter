
from enum import Enum

class GameState(Enum):
    GAME_OVER = 1
    ROBOT_TURN = 2
    HUMAN_TURN = 3
    INVALID = 4
    GAME_OVER_HUMAN = 5
    GAME_OVER_ROBOT = 6
