def middle_of_box(frame_shape, box):
    frame_width = frame_shape[0]
    frame_height = frame_shape[1]
    x_mid = (box[1] * frame_width + box[3] * frame_width) / 2
    y_mid = (box[0] * frame_height + box[2] * frame_height) / 2
    return int(x_mid), int(y_mid)
