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


def draw_circle(out_frame: NDArray, circle_number: int, point: Tuple[int, int],
                color: Tuple[int, int, int] = SOLID_BLUE):
    """
    Draws a circle in the given frame
    :param out_frame: The frame to draw a circle in
    :param circle_number: The number which is added to the circle
    :param point: The coordinates of the circle
    :param color: The color of the cirlce
    :return: A copy of the given frame with a cirlce drawn on it
    """
    out_frame = cv.putText(out_frame, str(circle_number), point, cv.FONT_HERSHEY_SIMPLEX, 0.7, SOLID_YELLOW, 2)
    return cv.circle(
        out_frame,
        point,
        10,
        color,
        2
    )


def add_horizontal_distance(out_frame: NDArray, distance: float, point: Tuple[int, int]):
    """
    Add the text for the horizontal distance to the given frame
    :param out_frame: The frame to write the text into
    :param distance: The horizontal distance between the person objects
    :param point: The coordinates of the text in the given frame
    :return: A copy of the frame with the text for the horizontal distance added
    """
    text = f'Estimated horizontal distance: {distance:.2f} cm' if distance != -1 else \
        "Estimated horizontal distance: N/A"
    return cv.putText(out_frame, text, point, cv.FONT_HERSHEY_SIMPLEX, 0.7, SOLID_WHITE, 2)


def add_distance_to_wall(out_frame: NDArray, securer_wall_distance: float, point: Tuple[int, int]):
    """
    Add the text for the distance between the securer and the climbing wall
    :param out_frame: The frame to write the text into
    :param securer_wall_distance: The distance between the securer and the wall
    :param point: The coordinates of the text in the frame
    :return: A copy of the frame with the text for the distance to wall calculation
    """
    text = f'Estimated distance to wall: {securer_wall_distance:.2f} cm' if securer_wall_distance != -1 else \
        "Estimated distance to wall: N/A"
    return cv.putText(out_frame, text, point, cv.FONT_HERSHEY_SIMPLEX, 0.7, SOLID_WHITE, 2)


def add_distance_to_fix_point(out_frame: NDArray, distance: float, point: Tuple[int, int]):
    """
    Add the text for the vertical distance to the latest passed fix point
    :param out_frame: The frame to add the text to
    :param distance: The calculated distance to the lastes fix point
    :param point: The coordinates of the text in the frame
    :return: A copy of the frame with the added text
    """
    text = f'Vertical distance to fixpoint: {distance:.2f} cm' if distance != -1 else \
        'Vertical distance to fixpoint: N/A'
    return cv.putText(out_frame, text, point, cv.FONT_HERSHEY_SIMPLEX, 0.7, SOLID_WHITE, 2)


class BirdViewWriter:
    """
    Provides the interface to write a "BirdView"-Output video.
    """

    def __init__(self, output_writer: OutputVideoWriter, frame_shape: Tuple[int, int]):
        """
        Constructor
        :param output_writer: The wrapped output writer
        :param frame_shape: The shape of the output frame
        """
        self.output_writer = output_writer
        self.frame_shape = frame_shape
        self.blank_image = np.zeros((frame_shape[1], frame_shape[0], 3), np.uint8)
        self.blank_image[:] = SOLID_BLACK_COLOR

    def __del__(self):
        """
        Destructor.
        Releases the wrapped output writer
        :return:
        """
        self.release()

    def write(self, person_positions: List, fix_points: List, horizontal_distance: float, securer_wall_distance: float,
              distance_to_fix_point: float) -> None:
        """
        Write the next birdview-frame with the information given:
        :param person_positions: The coordinates of the detected person objects in the analyzed frame
        and the climber
        :param fix_points: The coordinates of the fix points marked in the first frame
        :param horizontal_distance: The distance between the persons
        :param securer_wall_distance: The calculated distance of the securer to the climbing wall
        :param distance_to_fix_point: The calculated horizontal distance to the latest fix point
        """
        out_frame = np.copy(self.blank_image)
        for i, circle in enumerate(person_positions):
            out_frame = draw_circle(out_frame, i, circle)
        for i, fix_point in enumerate(fix_points):
            out_frame = draw_circle(out_frame, i, fix_point, SOLID_RED)
        out_frame = add_distance_to_fix_point(out_frame, distance_to_fix_point,
                                              (self.frame_shape[0] - 650, self.frame_shape[1] - 150))
        out_frame = add_horizontal_distance(out_frame, horizontal_distance, (self.frame_shape[0] - 650, self.frame_shape[1] - 125))
        out_frame = add_distance_to_wall(out_frame, securer_wall_distance,
                                         (self.frame_shape[0] - 650, self.frame_shape[1] - 100))
        self.output_writer.write(out_frame)

    def release(self) -> None:
        """
        Release the internal output video writer
        """
        self.output_writer.release()
