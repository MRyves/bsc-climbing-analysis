from typing import Tuple, List

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from risk import utils


def resize_with_aspect_ratio(image: NDArray, height: int, inter=cv.INTER_AREA) -> Tuple[NDArray, float]:
    """
    Since the image is likely to be too large to fit the screen of the users' computer it must be resized.
    This function resized the given image to have the given height.
    The resize-factor (r) is returned since it is used for mapping the coordinates of the resized image to the actual image.
    :param image: The image which should be resized
    :param height: The height to which the image should be resized, keeping the width / height ratio
    :param inter: Interpolation option
    :return: The resized image and the resize-factor
    """
    (image_h, image_w) = image.shape[:2]
    r = height / float(image_h)
    dimension = (int(image_w * r), height)
    print(f'r value: {r}')
    return cv.resize(image, dimension, interpolation=inter), r


class FixPointRisk:
    """
    Implements the fix-point distance risk.
    This class provides the interface to calculate the vertical distance to the latest passed fix point and the climber.
    """

    def __init__(self, frame_shape: Tuple[int, int], securer_height: int):
        """
        Constructor
        :param frame_shape: The shape of the image/frame in px (width, height)
        :param securer_height: The physical height of the securer in Centimeters
        """
        self.window_name = "MarkFixPoints"
        self.frame_shape = frame_shape
        self.securer_height = securer_height
        self.securer_height_frame = None
        self.fix_points_list = []
        self.fix_points = None
        self.fix_points_set = False
        self.init_frame = None
        self.resize_factor = None

    def analyze(self, frame: NDArray, person_boxes: NDArray) -> Tuple[List, float]:
        """
        Analyze the given frame and calculate the distance to the last passed fixpoint
        :param frame: The frame to be analyzed
        :param person_boxes: The detected person boxes in the given frame
        :return: The list of all fixpoints and the calculated distance to the last passed fixpoint in centimeters
        """
        distance_to_fix_point = -1
        if not self.fix_points_set:
            self.__init_fix_points(frame)
        if self.securer_height_frame is None and len(person_boxes) >= 1:
            self.securer_height_frame = (person_boxes[0][2] - person_boxes[0][0]) * self.frame_shape[1]
        if len(person_boxes) >= 2:
            climber_pos = utils.middle_of_box(self.frame_shape, person_boxes[1])
            closest_fix_point = self.__find_closest_fix_point(climber_pos)
            if closest_fix_point is not None:
                distance_to_fix_point = self.__calc_distance(closest_fix_point, climber_pos)
                print(f'Calculated distance to latest fix point: {distance_to_fix_point} cm')
        return self.fix_points_list, distance_to_fix_point

    def __find_closest_fix_point(self, climber_pos: Tuple[int, int]):
        """
        Find the last passed fixpoint with the current position of the climber
        :param climber_pos: The current position of the climber
        :return: The fixpoint closest fixpoint which is underneath the climber
        """
        reached_fix_points = self.fix_points[self.fix_points[:, 1] > climber_pos[1]]
        if len(reached_fix_points) > 0:
            closest_fix_point = reached_fix_points[reached_fix_points[:, 1].argmin()]
            return closest_fix_point.item(0), closest_fix_point.item(1)
        return None

    def __init_fix_points(self, frame: NDArray) -> None:
        """
        Shows the first frame so the user can mark the fixpoint in it
        :param frame: The frame to be shown to the user
        """
        # ask user to mark fix points in frame
        self.init_frame, self.resize_factor = resize_with_aspect_ratio(frame, 1000)
        cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback(self.window_name, self.__mark_fixpoint)
        while not self.fix_points_set:
            cv.imshow(self.window_name, self.init_frame)
            cv.waitKey(100)
        cv.destroyWindow(self.window_name)
        self.fix_points = np.array(self.fix_points_list)
        print(f'Successfully marked {len(self.fix_points_list)} fix points')

    def __mark_fixpoint(self, event, x, y, flags, param) -> None:
        """
        The callback function for marking the fixpoints in the first frame.
        The user may mark the fixpoints by clicking the left mouse key and confirm the selection by clicking the
        right mouse key
        :param event: The click event
        :param x: The x-coordinate of the click event
        :param y: The y-coordinate of the click event
        :param flags: (not used but required by the interface)
        :param param: (not used but required by the interface)
        """
        if event == cv.EVENT_LBUTTONDOWN:
            scaled_x = int(x / self.resize_factor)
            scaled_y = int(y / self.resize_factor)
            self.fix_points_list.append((scaled_x, scaled_y))
            cv.circle(self.init_frame, (x, y), 10, (0, 255, 255), 10)
        if event == cv.EVENT_RBUTTONDOWN:
            self.fix_points_set = True

    def __calc_distance(self, fix_point: Tuple[int, int], climber_pos: Tuple[int, int]) -> float:
        """
        Calculate the distance in centimeters using the physical height of the securer
        :param fix_point: The fixpoint to calculate the distance to
        :param climber_pos: The position of the climber
        :return: The calculated distance in centimeters
        """
        distance_frame = abs(fix_point[1] - climber_pos[1])
        return (distance_frame / self.securer_height_frame) * self.securer_height
