from enum import Enum, IntFlag

class GrowthPattern(IntFlag):
    BreadthFirst = 1
    DepthFirst = 2
    Slow = 4
    SlowBreadthFirst = BreadthFirst | Slow
    SlowDepthFirst = DepthFirst | Slow

    def __str__(self):
        return self.name