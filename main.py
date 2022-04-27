import argparse
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from numpy.typing import NDArray
from six import BytesIO

from risk.RiskAnalysis import RiskAnalysis
from VideoReader import VideoReader
from model import Model
from writer.OutputVideoWriter import OutputVideoWriter


def load_image_into_numpy_array(path: str) -> NDArray:
    """
    Loads image from given path and puts it into a numpy array.
    :param path: Path the the image
    :return: NDArray with shape (1, image_px_height, image_px_width, 3)
    """
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))
    (image_width, image_height) = image.size
    return np.array(image.getdata()).reshape(
        (1, image_height, image_width, 3)).astype(np.uint8)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start a climbing-video analysis.', prog='main.py',
                                     usage='%(prog)s <path-to-input-video> [options]',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', metavar='Input Video', type=str, help='The path to the input video.')
    parser.add_argument('-o', '--outputFolder', type=str, dest="output_folder", default='./output',
                        help='The output folder of the analysis result.')
    parser.add_argument('-n', '--outputVideoName', type=str, dest='output_video_name', default='analysis-output',
                        help='The name of the output video. This only changes the filename not the video format. The '
                             'format will always be .AVI')
    parser.add_argument('--fps', type=int, dest='fps', default=1,
                        help='How many frames per second should be analyzed of the input video.')
    parser.add_argument('--model', type=str, dest='model', default='Faster R-CNN ResNet152 V1 1024x1024',
                        help='Which TF model should be used to detect objects in the frames. See "model.py" for a '
                             'list of all available models.')
    parser.add_argument('--securerHeight', '-s', type=int, dest='securer_height', default=170,
                        help='The actual height of the securer in centimeters')
    parser.add_argument('--distanceWall', '-w', type=int, dest='distance_to_wall', default=50,
                        help='The actual distance of the securer to the climbing wall in the first frame')
    args = parser.parse_args()

    video_reader = VideoReader(args.input, args.fps)
    bird_view_2 = OutputVideoWriter(f'{args.output_folder}/{args.output_video_name}_birdview.avi',
                                    video_reader.consider_frames_per_second, video_reader.video_shape)
    analyzer = RiskAnalysis(video_reader.video_shape, bird_view_2, args.distance_to_wall, args.securer_height)
    video_writer = OutputVideoWriter(f'{args.output_folder}/{args.output_video_name}.avi',
                                     video_reader.consider_frames_per_second, video_reader.video_shape)
    model = Model(model_name=args.model)

    has_frame, frame = video_reader.next_frame()
    while has_frame:
        print(f'Starting analysis of frame {video_reader.current_frame}')
        starting_time = time.process_time()
        analyzed_frame, person_boxes = model.analyze(frame)
        analyzer.analyze(frame[0], person_boxes)
        elapsed_time = time.process_time() - starting_time
        print(f'Analyzed frame number: {str(video_reader.current_frame)} in {str(elapsed_time)}')
        video_writer.write(analyzed_frame)
        has_frame, frame = video_reader.next_frame()

    video_writer.release()
    analyzer.finished_analysis()
