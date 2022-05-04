from typing import Tuple, List

from numpy.typing import NDArray

from risk.HorizontalDistanceRisk import HorizontalDistanceRisk
from risk.FixPointRisk import FixPointRisk
from risk.WallDistanceRisk import WallDistanceRisk
from writer.BirdViewWriter import BirdViewWriter
from writer.OutputVideoWriter import OutputVideoWriter


def order_boxes(person_boxes: NDArray) -> NDArray:
    """
    Orders the given numpy array of person detection boxes by the y-axis value (then by x-axis value).
    This usually means that the climber is the first element and the securer the second element of the resulting array
    :param person_boxes: The detection boxes to be ordered
    :return: The ordered array of detection boxes
    """
    person_boxes = person_boxes[person_boxes[:, 2].argsort()]
    person_boxes = person_boxes[person_boxes[:, 1].argsort(kind='merge')]
    return person_boxes


class RiskAnalysis:
    """
    Class implementing the risk analysis part of the climbing situation
    """

    def __init__(self, frame_shape: Tuple[int, int], distance_to_wall: int, securer_height: int,
                 distance_camera_wall: int, output_writer: OutputVideoWriter = None):
        """
        Constructor
        :param frame_shape: the shape of the frame in pixels (width, height)
        :param distance_to_wall: The distance to the wall in the first frame. This value is used to determine the
        risk the securer being too far away from the climbing wall. Unit: centimeters
        :param securer_height: The real life height of the securer in centimeters
        :param distance_camera_wall: The real life distance of the camera to the climbing wall. Unit: centimeters
        :param output_writer: If given an output video of the analysis is written using the BirdViewWriter
        """
        self.output_writer = None
        self.frame_shape = frame_shape
        self.horizontal_distance_risk = HorizontalDistanceRisk(frame_shape, securer_height)
        self.wall_distance_risk = WallDistanceRisk(distance_to_wall, securer_height, distance_camera_wall)
        self.fix_point_risk = FixPointRisk(frame_shape, securer_height)
        if output_writer is not None:
            self.output_writer = BirdViewWriter(output_writer, self.frame_shape)

    def __del__(self) -> None:
        """
        Deconstruct: Finishes the analysis
        """
        self.finished_analysis()

    def analyze(self, frame: NDArray, person_boxes: NDArray) -> None:
        """
        Performs risk analysis on the given frame and logs the result on the console. It also writes to the output
        video writer if one is given.
        :param frame: The frame to be analyzed
        :param person_boxes: The bounding boxes of the detected person objects in the given frame
        """
        print(f'Risk analyzing a total of {len(person_boxes)} detection boxes...')
        person_boxes_ordered = order_boxes(person_boxes)
        box_centers, polygon_vertices, angle, distance = self.horizontal_distance_risk.calc_distance(
            person_boxes_ordered)
        securer_wall_distance = self.wall_distance_risk.calc_distance(person_boxes_ordered)
        fix_points, distance_to_fix_point = self.fix_point_risk.analyze(frame, person_boxes_ordered)
        print(f'Calculated angle: {angle} degrees -> Horizontal distance: {distance}')
        print(f'Estimated distance to wall: {securer_wall_distance} cm')
        self.write(box_centers, polygon_vertices, fix_points, angle, distance, securer_wall_distance,
                   distance_to_fix_point)

    def finished_analysis(self):
        """
        Release the internal output video writer if one was given in the constructor.
        """
        if self.has_output_writer:
            self.output_writer.release()

    def write(self, person_positions: List, polygon_edges: List, fix_points: List, angle: float, distance: float,
              securer_wall_distance: float, distance_to_fix_point: float) -> None:
        """
        Writes the analysis result to the output video writer
        :param person_positions: The position of the detected person in the frame
        :param polygon_edges: The edges for the triangle used to calculate the horizontal distance of the persons
        :param fix_points: The positions of the fix points marked by the user in the first frame
        :param angle: The angle in question for the horizontal distance measure
        :param distance: The estimated horizontal distance between the person objects
        :param securer_wall_distance: The calculated distance of the securer to the wall
        :param distance_to_fix_point: The calculated distance to the last passed fix point
        """
        if self.has_output_writer:
            self.output_writer.write(person_positions, polygon_edges, fix_points, angle, distance,
                                     securer_wall_distance, distance_to_fix_point)

    @property
    def has_output_writer(self) -> bool:
        """
        :return: true if an internal output video writer is given
        """
        return self.output_writer is not None
