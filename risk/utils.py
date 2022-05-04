from typing import Tuple

from numpy.typing import NDArray


def middle_of_box(frame_shape: Tuple[int, int], box: NDArray) -> Tuple[int, int]:
    """
    Calculates the coordinates of the middle of the bounding box in an image
    :param frame_shape: The shape of the frame in px (width, height)
    :param box: The bounding box
    :return: The coordinates of the middle of the box
    """
    frame_width = frame_shape[0]
    frame_height = frame_shape[1]
    x_mid = (box[1] * frame_width + box[3] * frame_width) / 2
    y_mid = (box[0] * frame_height + box[2] * frame_height) / 2
    return int(x_mid), int(y_mid)
