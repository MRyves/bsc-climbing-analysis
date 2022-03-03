import numpy as np
import tensorflow as tf
from PIL import Image
from six import BytesIO

from model import Model


def load_image_into_numpy_array(path):
    image = None
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))
    (image_width, image_height) = image.size
    return np.array(image.getdata()).reshape(
        (1, image_height, image_width, 3)).astype(np.uint8)


if __name__ == "__main__":
    model = Model()
    image_np = load_image_into_numpy_array('./resources/pictures/climbing1.png')
    analyzed_frame = model.analyze(image_np)
    img = Image.fromarray(analyzed_frame, 'RGB')
    img.save('results/climbing1.png')
    img.show()
