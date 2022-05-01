import math
from typing import List


class WallDistanceRisk:
    def __init__(self, securer_wall_distance: int = 50, physical_securer_height: int = 185,
                 camera_wall_distance: int = 470):
        self.securer_distance = securer_wall_distance
        self.camera_distance = camera_wall_distance
        self.securer_camera_distance = camera_wall_distance - securer_wall_distance
        self.physical_securer_height = physical_securer_height
        self.focal_length = None

    def calc_distance(self, person_boxes: List) -> float:
        if len(person_boxes) == 0:
            return -1
        securer_height = person_boxes[-1][2] - person_boxes[-1][0]
        if self.focal_length is None:
            self.focal_length = self.calc_focal_length(securer_height)
            return self.securer_distance
        securer_camera_distance_new = (self.physical_securer_height * self.focal_length) / securer_height
        return self.camera_distance - securer_camera_distance_new

    def calc_focal_length(self, securer_height: float) -> float:
        return (securer_height * self.securer_camera_distance) / self.physical_securer_height
