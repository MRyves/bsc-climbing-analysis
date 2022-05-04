from typing import Tuple

import cv2 as cv
import numpy as np
from numpy.typing import NDArray


class VideoReader:
    """
    Used to read a video frame by frame.
    """

    def __init__(self, video_path: str, consider_frames_per_second: int):
        """
        Constructor
        :param video_path: path to the video
        :param consider_frames_per_second: frames per second to be read, default is one frame per second
        """
        self.video_path = video_path
        self.consider_frames_per_second = consider_frames_per_second
        self.__cap = None
        self.frame_increment = None
        self.video_dimensions = {}
        self.current_frame = 1
        self.frames_read = 0

    def next_frame(self) -> tuple[bool, NDArray]:
        """
        Read the next frame
        :return: tuple(
            has_frame: boolean if next frame was read successfully, this is false if end of video is reached <br>
            frame: the frame as a NDArray
        )
        """
        self.__update_current_frame()
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.current_frame)
        has_frame, frame = self.cap.read()
        if has_frame:
            self.frames_read += 1
            frame = frame.reshape((1, self.video_dimensions['height'], self.video_dimensions['width'], 3)) \
                .astype(np.uint8)
            return has_frame, frame
        else:
            return has_frame, None

    def __init_cap(self) -> None:
        """
        Initializes the internal video reader if it wasn't initialized yet.
        """
        if self.__cap is None:
            self.__cap = cv.VideoCapture(self.video_path)
            self.video_dimensions['height'] = int(self.__cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            self.video_dimensions['width'] = int(self.__cap.get(cv.CAP_PROP_FRAME_WIDTH))
            self.video_dimensions['fps'] = int(self.__cap.get(cv.CAP_PROP_FPS))
            self.frame_increment = int(self.video_dimensions['fps'] / self.consider_frames_per_second)
            print(
                f'Initialized video capturer for video "{self.video_path}" \n with fps: '
                f'{self.video_dimensions["fps"]} \n capturer will increment frame by: {self.frame_increment}')

    @property
    def cap(self):
        """
        Initialize the video capturer if needed
        :return: The video capturer instance
        """
        self.__init_cap()
        return self.__cap

    @property
    def video_shape(self) -> Tuple[int, int]:
        """
        :return: The shape of the video in pixels (width, height)
        """
        self.__init_cap()
        return self.video_dimensions['width'], self.video_dimensions['height']

    @property
    def video_fps(self) -> int:
        """
        :return: The fps of the video which is read
        """
        self.__init_cap()
        return self.video_dimensions['fps']

    def __update_current_frame(self) -> None:
        """
        Updates the current_frame by incrementing it with the calculated frame_increment
        """
        if self.frames_read > 0:
            self.current_frame += self.frame_increment
