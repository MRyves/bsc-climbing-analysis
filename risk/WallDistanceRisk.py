import math
from typing import List


def calc_tangent_extension(a, b, c):
    return (b * c / a) - b


class WallDistanceRisk:
    def __init__(self, frame_height: int, initial_wall_distance: int = 80):
        self.frame_height = frame_height
        self.initial_securer_wall_distance = initial_wall_distance
        self.initial_securer_height = None

    def calc_distance(self, person_boxes: List) -> float:
        if len(person_boxes) != 2:
            return -1
        securer_height = abs(person_boxes[1][0] - person_boxes[1][2]) * self.frame_height
        if self.initial_securer_height is None:
            self.initial_securer_height = securer_height
            return self.initial_securer_wall_distance
        d = calc_tangent_extension(self.initial_securer_height, self.init_tangent_length, securer_height)
        return math.sqrt((self.init_tangent_length + d) ** 2 - (securer_height / 2) ** 2)

    @property
    def init_tangent_length(self):
        if self.initial_securer_height is None:
            return None
        return math.sqrt(self.initial_securer_wall_distance ** 2 + (self.initial_securer_height / 2) ** 2)
