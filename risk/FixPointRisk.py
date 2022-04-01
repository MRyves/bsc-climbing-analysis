import cv2 as cv
import numpy as np

from risk import utils


def resize_with_aspect_ratio(image, height, inter=cv.INTER_AREA):
    (image_h, image_w) = image.shape[:2]
    r = height / float(image_h)
    dimension = (int(image_w * r), height)
    print(f'r value: {r}')
    return cv.resize(image, dimension, interpolation=inter), r


class FixPointRisk:
    def __init__(self, frame_shape, securer_height=170):
        self.window_name = "MarkFixPoints"
        self.frame_shape = frame_shape
        self.securer_height = securer_height
        self.securer_height_frame = None
        self.fix_points_list = []
        self.fix_points = None
        self.fix_points_set = False
        self.init_frame = None
        self.resize_factor = None

    def analyze(self, frame, person_boxes):
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

    def __find_closest_fix_point(self, climber_pos):
        reached_fix_points = self.fix_points[self.fix_points[:, 1] > climber_pos[1]]
        if len(reached_fix_points) > 0:
            closest_fix_point = reached_fix_points[reached_fix_points[:, 1].argmin()]
            return closest_fix_point.item(0), closest_fix_point.item(1)
        return None

    def __init_fix_points(self, frame):
        # ask user to mark fix points in frame
        self.init_frame, self.resize_factor = resize_with_aspect_ratio(frame, 1000)
        cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback(self.window_name, self.get_fix_points)
        while not self.fix_points_set:
            cv.imshow(self.window_name, self.init_frame)
            cv.waitKey(100)
        cv.destroyWindow(self.window_name)
        self.fix_points = np.array(self.fix_points_list)
        print(f'Successfully marked {len(self.fix_points_list)} fix points')

    def get_fix_points(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            scaled_x = int(x / self.resize_factor)
            scaled_y = int(y / self.resize_factor)
            self.fix_points_list.append((scaled_x, scaled_y))
            cv.circle(self.init_frame, (x, y), 10, (0, 255, 255), 10)
        if event == cv.EVENT_RBUTTONDOWN:
            self.fix_points_set = True

    def __calc_distance(self, fix_point, climber_pos):
        distance_frame = abs(fix_point[1] - climber_pos[1])
        return (distance_frame / self.securer_height_frame) * self.securer_height
