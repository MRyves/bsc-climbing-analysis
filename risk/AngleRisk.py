import math
import operator

import numpy as np

from risk import utils


def calc_angle(polygon_vertices, right_shifted):
    if len(polygon_vertices) != 3:
        return -1

    a = polygon_vertices[0]
    b = polygon_vertices[2]
    c = polygon_vertices[1]
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    if right_shifted:
        return ang + 360 if ang < 0 else ang
    else:
        return abs(ang)


class AngleRisk:
    def __init__(self, frame_shape, angle_threshold=90):
        self.frame_shape = frame_shape
        self.angle_threshold = angle_threshold

    def identify(self, person_boxes):
        box_centers = [utils.middle_of_box(self.frame_shape, box) for box in person_boxes]
        was_right_shifted, polygon_vertices = self.__calc_polygon_vertices(person_boxes)
        angle = calc_angle(polygon_vertices, was_right_shifted)
        return box_centers, polygon_vertices, angle

    def __calc_polygon_vertices(self, person_boxes: np.ndarray):
        if len(person_boxes) != 2:
            # As of right now, there is no way to draw a polygon if there are more persons detected than two
            # Therefore skipping the calculation in that case
            return False, []
        # Shift first person box (the securer) to the side for height of person
        securer_height = person_boxes[0][2] - person_boxes[0][0]
        shift_operator = self.__get_shift_operator(person_boxes[0], person_boxes[1])
        shifted_securer = [person_boxes[0][0], shift_operator(person_boxes[0][1], securer_height), person_boxes[0][2],
                           shift_operator(person_boxes[0][3], securer_height)]
        person_boxes = np.append(person_boxes, [shifted_securer], axis=0)
        return (shift_operator == operator.add), [utils.middle_of_box(self.frame_shape, box) for box in person_boxes]

    @staticmethod
    def __get_shift_operator(securer, climber):
        xmin_securer = securer[1]
        xmin_climber = climber[1]
        if xmin_securer <= xmin_climber:
            return operator.add
        else:
            return operator.sub
