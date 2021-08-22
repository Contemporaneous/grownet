from enum import Enum, IntFlag

class GrowthPattern(IntFlag):
    BreadthFirst = 1
    DepthFirst = 2
    Slow = 4
    Fast = 8
    SlowBreadthFirst = BreadthFirst | Slow
    SlowDepthFirst = DepthFirst | Slow
    FastBreadthFirst = BreadthFirst | Fast
    FastDepthFirst = DepthFirst | Fast

    def __str__(self):
        return self.name