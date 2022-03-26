import cv2 as cv
import numpy as np

from writer.OutputVideoWriter import OutputVideoWriter

SOLID_BLACK_COLOR = (41, 41, 41)
SOLID_YELLOW = (255, 255, 0)
AVI_FORMAT = cv.VideoWriter_fourcc(*"MJPG")


def draw_circle(out_frame, circle_number, point):
    out_frame = cv.putText(out_frame, str(circle_number), point, cv.FONT_HERSHEY_SIMPLEX, 0.7, SOLID_YELLOW, 2)
    return cv.circle(
        out_frame,
        point,
        10,
        (192, 133, 156),
        2
    )


def draw_polygon(out_frame, polygon_vertices):
    if len(polygon_vertices) == 3:
        polygon_vertices = np.array(polygon_vertices, np.int32)
        polygon_vertices = polygon_vertices.reshape((-1, 1, 2))
        out_frame = cv.polylines(out_frame, [polygon_vertices], True, color=(0, 0, 255), thickness=5)
    return out_frame


def add_angle(out_frame, angle, point):
    text = f'Angle: {angle:.2f} Deg' if angle != -1 else "Angle: N/A"
    return cv.putText(out_frame, text, point, cv.FONT_HERSHEY_SIMPLEX, 0.7, SOLID_YELLOW, 2)


def add_wall_distance(out_frame, securer_wall_distance, point):
    text = f'Estimated distance to wall: {securer_wall_distance:.2f} cm' if securer_wall_distance != -1 else \
        "Estimated distance to wall: N/A"
    return cv.putText(out_frame, text, point, cv.FONT_HERSHEY_SIMPLEX, 0.7, SOLID_YELLOW, 2)


class BirdViewWriter:
    def __init__(self, output_writer: OutputVideoWriter, frame_shape: tuple):
        self.output_writer = output_writer
        self.frame_shape = frame_shape
        self.blank_image = np.zeros((frame_shape[1], frame_shape[0], 3), np.uint8)
        self.blank_image[:] = SOLID_BLACK_COLOR

    def __del__(self):
        self.release()

    def write(self, circles, polygon_vertices, angle, securer_wall_distance):
        out_frame = np.copy(self.blank_image)
        for i, circle in enumerate(circles):
            out_frame = draw_circle(out_frame, i, circle)
        out_frame = draw_polygon(out_frame, polygon_vertices)
        out_frame = add_angle(out_frame, angle, (self.frame_shape[0] - 250, self.frame_shape[1] - 100))
        out_frame = add_wall_distance(out_frame, securer_wall_distance,
                                      (self.frame_shape[0] - 450, self.frame_shape[1] - 50))
        self.output_writer.write(out_frame)

    def release(self):
        self.output_writer.release()
