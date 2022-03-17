import cv2
import numpy as np


class VideoReader:
    def __init__(self, video_path, consider_frames_per_second=1):
        self.video_path = video_path
        self.consider_frames_per_second = consider_frames_per_second
        self.__cap = None
        self.frame_increment = None
        self.video_dimensions = {}
        self.current_frame = 1
        self.frames_read = 0

    def next_frame(self):
        self.__update_current_frame()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        has_frame, frame = self.cap.read()
        if has_frame:
            self.frames_read += 1
            frame = frame.reshape((1, self.video_dimensions['height'], self.video_dimensions['width'], 3)) \
                .astype(np.uint8)
            return has_frame, frame
        else:
            return has_frame, None

    def __init_cap(self):
        if self.__cap is None:
            self.__cap = cv2.VideoCapture(self.video_path)
            self.video_dimensions['height'] = int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_dimensions['width'] = int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_dimensions['fps'] = int(self.__cap.get(cv2.CAP_PROP_FPS))
            self.frame_increment = int(self.video_dimensions['fps'] / self.consider_frames_per_second)
            print(
                f'Initialized video capturer for video "{self.video_path}" \n with fps: '
                f'{self.video_dimensions["fps"]} \n capturer will increment frame by: {self.frame_increment}')

    @property
    def cap(self):
        self.__init_cap()
        return self.__cap

    @property
    def video_shape(self):
        self.__init_cap()
        return self.video_dimensions['width'], self.video_dimensions['height']

    @property
    def video_fps(self):
        self.__init_cap()
        return self.video_dimensions['fps']

    def __update_current_frame(self):
        if self.frames_read > 0:
            self.current_frame += self.frame_increment

