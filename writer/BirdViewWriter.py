import numpy as np
from cv2 import VideoWriter_fourcc, circle

from writer.OutputVideoWriter import OutputVideoWriter

SOLID_BLACK_COLOR = (41, 41, 41)
AVI_FORMAT = VideoWriter_fourcc(*"MJPG")


def draw_circle(out_frame, point):
    return circle(
        out_frame,
        point,
        10,
        (192, 133, 156),
        20
    )


class BirdViewWriter:
    def __init__(self, output_path, fps, shape):
        self.output_path = output_path
        self.fps = fps
        self.frame_width = shape[0]
        self.frame_height = shape[1]
        self.blank_image = np.zeros((self.frame_height, self.frame_width, 3), np.uint8)
        self.blank_image[:] = SOLID_BLACK_COLOR
        self.current_frame = 1
        self.writer = OutputVideoWriter(self.output_path, self.fps, shape)

    def __del__(self):
        self.writer.release()

    def digest(self, person_boxes):
        out_frame = np.copy(self.blank_image)
        if len(person_boxes) > 0:
            print(f'Digesting a total of {len(person_boxes)} person detections')
            for person_box in person_boxes:
                x, y = self.middle_of_box(person_box)
                out_frame = draw_circle(out_frame, (x, y))

        self.writer.write(out_frame)
        self.current_frame += 1

    def release(self):
        self.__del__()

    def middle_of_box(self, box):
        x_mid = (box[1] * self.frame_width + box[3] * self.frame_width) / 2
        y_mid = (box[0] * self.frame_height + box[2] * self.frame_height) / 2
        return int(x_mid), int(y_mid)
