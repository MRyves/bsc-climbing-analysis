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
    def __init__(self, frame_shape):
        self.window_name = "MarkFixPoints"
        self.frame_shape = frame_shape
        self.fix_points_list = []
        self.fix_points = None
        self.fix_points_set = False
        self.init_frame = None
        self.resize_factor = None

    def analyze(self, frame, person_boxes):
        if not self.fix_points_set:
            self.__init_fix_points(frame)
        climber_pos = utils.middle_of_box(self.frame_shape, person_boxes[1])
        closest_fix_point = self.__find_closest_fix_point(climber_pos)
        print(f'Climber position: {climber_pos}')
        print(f'Closest fix point: {closest_fix_point}')
        return

    def __find_closest_fix_point(self, climber_pos):
        closest_fix_point = None
        reached_fix_points = self.fix_points[self.fix_points[:, 1] > climber_pos[1]]
        if len(reached_fix_points) > 0:
            closest_fix_point = reached_fix_points[reached_fix_points[:, 1].argmin()]
        return closest_fix_point

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
            print(f'Caught mouse click: {(x, y)}')
            scaled_x = int(x / self.resize_factor)
            scaled_y = int(y / self.resize_factor)
            self.fix_points_list.append((scaled_x, scaled_y))
            cv.circle(self.init_frame, (x, y), 10, (0, 255, 255), 10)
        if event == cv.EVENT_RBUTTONDOWN:
            self.fix_points_set = True
