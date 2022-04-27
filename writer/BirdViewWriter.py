from typing import Tuple, List

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from writer.OutputVideoWriter import OutputVideoWriter

SOLID_BLACK_COLOR = (41, 41, 41)
SOLID_YELLOW = (0, 255, 255)
SOLID_WHITE = (255, 255, 255)
SOLID_RED = (0, 0, 255)
SOLID_BLUE = (255, 0, 0)
SOLID_TURQUOISE = (255, 255, 175)
AVI_FORMAT = cv.VideoWriter_fourcc(*"MJPG")


def draw_circle(out_frame, circle_number, point, color=SOLID_BLUE):
    out_frame = cv.putText(out_frame, str(circle_number), point, cv.FONT_HERSHEY_SIMPLEX, 0.7, SOLID_YELLOW, 2)
    return cv.circle(
        out_frame,
        point,
        10,
        color,
        2
    )


def draw_polygon(out_frame: NDArray, polygon_vertices: List):
    if len(polygon_vertices) == 3:
        polygon_vertices = np.array(polygon_vertices, np.int32)
        polygon_vertices = polygon_vertices.reshape((-1, 1, 2))
        out_frame = cv.polylines(out_frame, [polygon_vertices], True, color=SOLID_TURQUOISE, thickness=2)
    return out_frame


def add_angle(out_frame, angle, point):
    text = f'Angle: {angle:.2f} Deg' if angle != -1 else "Angle: N/A"
    return cv.putText(out_frame, text, point, cv.FONT_HERSHEY_SIMPLEX, 0.7, SOLID_WHITE, 2)


def add_distance_to_wall(out_frame, securer_wall_distance, point):
    text = f'Estimated distance to wall: {securer_wall_distance:.2f} cm' if securer_wall_distance != -1 else \
        "Estimated distance to wall: N/A"
    return cv.putText(out_frame, text, point, cv.FONT_HERSHEY_SIMPLEX, 0.7, SOLID_WHITE, 2)


def add_distance_to_fix_point(out_frame, distance, point):
    text = f'Vertical distance to fixpoint: {distance:.2f} cm' if distance != -1 else \
        'Vertical distance to fixpoint: N/A'
    return cv.putText(out_frame, text, point, cv.FONT_HERSHEY_SIMPLEX, 0.7, SOLID_WHITE, 2)


class BirdViewWriter:
    """
    Provides the interface to write a "BirdView"-Output video.
    """

    def __init__(self, output_writer: OutputVideoWriter, frame_shape: Tuple[int, int]):
        self.output_writer = output_writer
        self.frame_shape = frame_shape
        self.blank_image = np.zeros((frame_shape[1], frame_shape[0], 3), np.uint8)
        self.blank_image[:] = SOLID_BLACK_COLOR

    def __del__(self):
        self.release()

    def write(self, circles: List, polygon_vertices: List, fix_points: List, angle: float, securer_wall_distance: float,
              distance_to_fix_point: float) -> None:
        """
        Write the next birdview-frame with the information given:
        :param circles: The coordinates of the detected person objects in the analyzed frame
        :param polygon_vertices: The vertices of the polygon used to measure the horizontal distance of the securer
        and the climber
        :param fix_points: The coordinates of the fix points marked in the first frame
        :param angle: The angle of the corner of the drawn polygon to measure horizontal distance
        :param securer_wall_distance: The calculated distance of the securer to the climbing wall
        :param distance_to_fix_point: The calculated horizontal distance to the latest fix point
        """
        out_frame = np.copy(self.blank_image)
        for i, circle in enumerate(circles):
            out_frame = draw_circle(out_frame, i, circle)
        for i, fix_point in enumerate(fix_points):
            out_frame = draw_circle(out_frame, i, fix_point, SOLID_RED)
        out_frame = draw_polygon(out_frame, polygon_vertices)
        out_frame = add_distance_to_fix_point(out_frame, distance_to_fix_point,
                                              (self.frame_shape[0] - 650, self.frame_shape[1] - 150))
        out_frame = add_angle(out_frame, angle, (self.frame_shape[0] - 650, self.frame_shape[1] - 125))
        out_frame = add_distance_to_wall(out_frame, securer_wall_distance,
                                         (self.frame_shape[0] - 650, self.frame_shape[1] - 100))
        self.output_writer.write(out_frame)

    def release(self):
        self.output_writer.release()
