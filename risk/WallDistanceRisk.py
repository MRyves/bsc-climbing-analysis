from typing import List

from numpy.typing import NDArray


class WallDistanceRisk:
    """
    This class calculates the distance of the securer to the climbing wall.
    It makes use of the focal-length formula.
    """

    def __init__(self, securer_wall_distance: int, physical_securer_height: int,
                 camera_wall_distance: int):
        """
        Constructor
        :param securer_wall_distance: the actual distance between securer and the wall in the first frame. Unit:
        Centimiters
        :param physical_securer_height: The actual (physical) height of the securing person. Unit: Centimeters
        :param camera_wall_distance: The actual distance between the camera and the wall: Unit Centimeters
        """
        self.securer_distance = securer_wall_distance
        self.camera_distance = camera_wall_distance
        self.securer_camera_distance = camera_wall_distance - securer_wall_distance
        self.physical_securer_height = physical_securer_height
        self.focal_length = None

    def calc_distance(self, person_boxes: NDArray) -> float:
        """
        Calculate the distance using the initial values provided to the constructor.
        :param person_boxes: The detected person objects, it is assumed that the securer is always the last item in
        this list
        :return: The calculated distance in Centimeters
        """
        if len(person_boxes) != 2:
            return -1
        securer_height = person_boxes[-1][2] - person_boxes[-1][0]
        if self.focal_length is None:
            self.focal_length = self.calc_focal_length(securer_height)
            return self.securer_distance
        securer_camera_distance_new = (self.physical_securer_height * self.focal_length) / securer_height
        return max(self.camera_distance - securer_camera_distance_new, 0)

    def calc_focal_length(self, securer_height: float) -> float:
        """
        Calculate the focal length of the camera-lens
        :param securer_height: the height of the securer in pixels
        :return: The calculated focal length
        """
        return (securer_height * self.securer_camera_distance) / self.physical_securer_height
