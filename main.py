import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc
from six import BytesIO

from BirdViewWriter import BirdViewWriter
from VideoReader import VideoReader
from model import Model

VIDEO_NAME = 'VID_20220309_212145'
VIDEO_FORMAT = 'mp4'
VIDEO_SUB_FOLDER = 'second_batch'


def load_image_into_numpy_array(path):
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))
    (image_width, image_height) = image.size
    return np.array(image.getdata()).reshape(
        (1, image_height, image_width, 3)).astype(np.uint8)


if __name__ == "__main__":
    video_reader = VideoReader(f'resources/videos/{VIDEO_SUB_FOLDER}/{VIDEO_NAME}.{VIDEO_FORMAT}', 1)
    bird_view = BirdViewWriter(video_reader.video_shape[0], video_reader.video_shape[1],
                               video_reader.consider_frames_per_second,
                               f'./results/videos/{VIDEO_SUB_FOLDER}/{VIDEO_NAME}_birdview.avi')
    video_format = VideoWriter_fourcc(*'MJPG')
    video_writer = VideoWriter(f'./results/videos/{VIDEO_SUB_FOLDER}/{VIDEO_NAME}.avi', video_format,
                               video_reader.consider_frames_per_second,
                               video_reader.video_shape)
    model = Model(model_name='Faster R-CNN ResNet152 V1 1024x1024')
    has_frame, frame = video_reader.next_frame()
    while has_frame:
        print(f'Starting analysis of frame {video_reader.current_frame}')
        starting_time = time.process_time()
        analyzed_frame, person_boxes = model.analyze(frame)
        bird_view.digest(person_boxes)
        elapsed_time = time.process_time() - starting_time
        print(f'Analyzed frame number: {str(video_reader.current_frame)} in {str(elapsed_time)}')
        analyzed_frame = cv2.resize(analyzed_frame, video_reader.video_shape)
        video_writer.write(analyzed_frame)
        has_frame, frame = video_reader.next_frame()

    video_writer.release()
    bird_view.release()
