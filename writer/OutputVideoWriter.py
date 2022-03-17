from cv2 import VideoWriter, VideoWriter_fourcc, resize

VIDEO_FORMAT = VideoWriter_fourcc(*'MJPG')


class OutputVideoWriter:
    def __init__(self, output_path, fps, shape):
        self.output_path = output_path
        self.fps = fps
        self.shape = shape
        self.__writer = None

    def __del__(self):
        if self.__writer is not None:
            self.__writer.release()
            self.__writer = None

    def write(self, frame):
        resized_frame = resize(frame, self.shape)
        self.writer.write(resized_frame)

    def release(self):
        self.__del__()

    @property
    def writer(self):
        if self.__writer is None:
            self.__writer = VideoWriter(self.output_path, VIDEO_FORMAT, self.fps, self.shape)
        return self.__writer
