import math

import numpy as np

from writer.BirdViewWriter import BirdViewWriter


def order_boxes(person_boxes):
    person_boxes = person_boxes[person_boxes[:, 2].argsort()]
    person_boxes = person_boxes[person_boxes[:, 1].argsort(kind='merge')]
    return person_boxes


def calc_angle(polygon_vertices):
    if len(polygon_vertices) == 3:
        a = polygon_vertices[0]
        b = polygon_vertices[2]
        c = polygon_vertices[1]
        ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        return ang + 360 if ang < 0 else ang
    else:
        return -1


def calc_d(a, b, c):
    return (b * c / a) - b


class RiskAnalysis:
    def __init__(self, frame_shape, output_writer=None):
        self.output_writer = None
        self.frame_shape = frame_shape
        self.initial_securer_height = None
        self.initial_securer_wall_distance = 80
        if output_writer is not None:
            self.output_writer = BirdViewWriter(output_writer, self.frame_shape)

    def __del__(self):
        self.finished_analysis()

    def analyze(self, person_boxes):
        print(f'Risk analyzing a total of {len(person_boxes)} detection boxes...')
        person_boxes = order_boxes(person_boxes)
        box_centers = [self.__middle_of_box(box) for box in person_boxes]
        polygon_vertices = self.__calc_polygon_vertices(person_boxes)
        angle = calc_angle(polygon_vertices)
        securer_wall_distance = self.calc_distance(person_boxes)
        print(f'Calculated angle: {angle} degrees')
        print(f'Estimated distance to wall: {securer_wall_distance} cm')
        self.write(box_centers, polygon_vertices, angle, securer_wall_distance)

    def calc_distance(self, person_boxes):
        if len(person_boxes) != 2:
            return -1
        securer_height = abs(person_boxes[1][0] - person_boxes[1][2]) * self.frame_height
        if self.initial_securer_height is None:
            self.initial_securer_height = securer_height
            return self.initial_securer_wall_distance
        d = calc_d(self.initial_securer_height, self.init_b_distance, securer_height)
        return math.sqrt((self.init_b_distance + d) ** 2 - (securer_height / 2) ** 2)

    def finished_analysis(self):
        if self.has_output_writer:
            self.output_writer.release()

    def write(self, circles, polygon_edges, angle, securer_wall_distance):
        if self.has_output_writer:
            self.output_writer.write(circles, polygon_edges, angle, securer_wall_distance)

    def __calc_polygon_vertices(self, person_boxes: np.ndarray):
        if len(person_boxes) != 2:
            # As of right now, there is no way to draw a polygon if there are more persons detected than two
            # Therefore skipping the calculation in that case
            return []
        # Shift first person box (the securer) to the right for height of person
        securer_height = person_boxes[0][2] - person_boxes[0][0]
        shifted_securer = [person_boxes[0][0], person_boxes[0][1] + securer_height, person_boxes[0][2],
                           person_boxes[0][3] + securer_height]
        person_boxes = np.append(person_boxes, [shifted_securer], axis=0)
        return [self.__middle_of_box(box) for box in person_boxes]

    def __middle_of_box(self, box):
        x_mid = (box[1] * self.frame_width + box[3] * self.frame_width) / 2
        y_mid = (box[0] * self.frame_height + box[2] * self.frame_height) / 2
        return int(x_mid), int(y_mid)

    @property
    def init_b_distance(self):
        if self.initial_securer_height is None:
            return None
        return math.sqrt(self.initial_securer_wall_distance ** 2 + (self.initial_securer_height / 2) ** 2)

    @property
    def has_output_writer(self):
        return self.output_writer is not None

    @property
    def frame_width(self):
        return self.frame_shape[0]

    @property
    def frame_height(self):
        return self.frame_shape[1]
