from typing import Tuple
from cv2 import VideoWriter, VideoWriter_fourcc, resize
from numpy.typing import NDArray

VIDEO_FORMAT = VideoWriter_fourcc(*'MJPG')


class OutputVideoWriter:
    """
    Provides the interface to writing single frames to an output video in the '.avi' format.
    """

    def __init__(self, output_path: str, fps: int, shape: Tuple[int, int]):
        """
        Constructor
        :param output_path: the path of the output video
        :param fps: the fps of the output video
        :param shape: the video dimensions in pixels (width, height)
        """
        self.output_path = output_path
        self.fps = fps
        self.shape = shape
        self.__writer = None

    def __del__(self) -> None:
        """
        Deconstruct.
        Releases the internal video writer and sets it to None
        """
        if self.__writer is not None:
            self.__writer.release()
            self.__writer = None

    def write(self, frame: NDArray) -> None:
        """
        Write given frame to the output video
        :param frame: Image to add to the output video
        """
        resized_frame = resize(frame, self.shape)
        self.writer.write(resized_frame)

    def release(self) -> None:
        """
        Release the internal video writer
        """
        self.__del__()

    @property
    def writer(self):
        """
        Initialized the internal video writer if not already done so
        :return: The video writer instance
        """
        if self.__writer is None:
            self.__writer = VideoWriter(self.output_path, VIDEO_FORMAT, self.fps, self.shape)
        return self.__writer
