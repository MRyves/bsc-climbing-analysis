import numpy as np

from cv2 import VideoWriter_fourcc, VideoWriter, circle

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
    def __init__(self, width, height, fps, output_path):
        self.frame_width = int(width)
        self.frame_height = int(height)
        self.fps = int(fps)
        self.output_path = output_path
        self.blank_image = np.zeros((self.frame_height, self.frame_width, 3), np.uint8)
        self.blank_image[:] = SOLID_BLACK_COLOR
        self.__video_writer = None
        self.current_frame = 1

    def __del__(self):
        if self.__video_writer is not None:
            self.__video_writer.release()
            self.__video_writer = None

    def digest(self, person_boxes):
        out_frame = np.copy(self.blank_image)
        if len(person_boxes) > 0:
            print('Digesting a total of {} person detections'.format(len(person_boxes)))
            for i in range(len(person_boxes)):
                x, y = self.middle_of_box(person_boxes[i])
                out_frame = draw_circle(out_frame, (x, y))

        self.video_writer.write(out_frame)
        self.current_frame += 1

    def release(self):
        if self.__video_writer is not None:
            self.__video_writer.release()
            self.__video_writer = None

    @property
    def video_writer(self):
        if self.__video_writer is None:
            self.__video_writer = VideoWriter(self.output_path, AVI_FORMAT, self.fps,
                                              (self.frame_width, self.frame_height))
        return self.__video_writer

    def middle_of_box(self, box):
        x_mid = (box[1] * self.frame_width + box[3] * self.frame_width) / 2
        y_mid = (box[0] * self.frame_height + box[2] * self.frame_height) / 2
        return int(x_mid), int(y_mid)
