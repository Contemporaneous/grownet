from enum import Enum

class GrowthPattern(Enum):
    BreadthFirst = 1
    DepthFirst = 2

    def __str__(self):
        return self.name