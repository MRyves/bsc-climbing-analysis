from typing import Tuple, List

from numpy.typing import NDArray

from risk import utils


class HorizontalDistanceRisk:
    """
    Provides the interface to calculate the horizontal risk between the climber and the securer.
    """
    def __init__(self, frame_shape: Tuple[int, int], securer_height: int):
        """
        Constructor
        :param frame_shape: The shape of the image in px (width, height)
        :param securer_height: The physical height of the securer
        """
        self.frame_shape = frame_shape
        self.securer_height = securer_height

    def calc_distance(self, person_boxes: NDArray) -> Tuple[List, float]:
        """
        Calculate the horizontal distance between the two identified person objects
        :param person_boxes: The bounding boxes of the detected person objects
        :return: The centers of the bounding boxes and the calculated distance
        """
        if len(person_boxes) != 2:
            return [], -1
        box_centers = [utils.middle_of_box(self.frame_shape, box) for box in person_boxes]
        securer_height_px = abs(person_boxes[0][2] - person_boxes[0][0]) * self.frame_shape[1]
        distance = self.__calc_horizontal_distance(box_centers, securer_height_px)
        return box_centers, distance

    def __calc_horizontal_distance(self, box_centers: List[Tuple[int, int]], securer_height_px: float) -> float:
        """
        Calculate the horizontal distance using the physical height of the securer as a reference
        :param box_centers: The centers of the detected person objects
        :param securer_height_px: The height of the securer in px
        :return: The distance between the two persons
        """
        distance_px = abs(box_centers[0][0] - box_centers[1][0])
        return (distance_px / securer_height_px) * self.securer_height
