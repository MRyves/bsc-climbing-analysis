import math

import numpy as np

from risk.AngleRisk import AngleRisk
from risk.FixPointRisk import FixPointRisk
from risk.WallDistanceRisk import WallDistanceRisk
from writer.BirdViewWriter import BirdViewWriter


def order_boxes(person_boxes):
    person_boxes = person_boxes[person_boxes[:, 2].argsort()]
    person_boxes = person_boxes[person_boxes[:, 1].argsort(kind='merge')]
    return person_boxes


class RiskAnalysis:
    def __init__(self, frame_shape, output_writer=None):
        self.output_writer = None
        self.frame_shape = frame_shape
        self.angle_risk = AngleRisk(frame_shape)
        self.wall_distance_risk = WallDistanceRisk(self.frame_shape[1])
        self.fix_point_risk = FixPointRisk(frame_shape)
        if output_writer is not None:
            self.output_writer = BirdViewWriter(output_writer, self.frame_shape)

    def __del__(self):
        self.finished_analysis()

    def analyze(self, frame, person_boxes):
        print(f'Risk analyzing a total of {len(person_boxes)} detection boxes...')
        person_boxes = order_boxes(person_boxes)
        box_centers, polygon_vertices, angle = self.angle_risk.identify(person_boxes)
        securer_wall_distance = self.wall_distance_risk.calc_distance(person_boxes)
        self.fix_point_risk.analyze(frame, person_boxes)
        print(f'Calculated angle: {angle} degrees')
        print(f'Estimated distance to wall: {securer_wall_distance} cm')
        self.write(box_centers, polygon_vertices, angle, securer_wall_distance)

    def finished_analysis(self):
        if self.has_output_writer:
            self.output_writer.release()

    def write(self, circles, polygon_edges, angle, securer_wall_distance):
        if self.has_output_writer:
            self.output_writer.write(circles, polygon_edges, angle, securer_wall_distance)

    @property
    def has_output_writer(self):
        return self.output_writer is not None
