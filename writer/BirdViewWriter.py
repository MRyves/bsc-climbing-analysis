import cv2 as cv
import numpy as np

from writer.OutputVideoWriter import OutputVideoWriter

SOLID_BLACK_COLOR = (41, 41, 41)
SOLID_YELLOW = (255, 255, 0)
AVI_FORMAT = cv.VideoWriter_fourcc(*"MJPG")


def draw_circle(out_frame, circle_number, point):
    out_frame = cv.putText(out_frame, str(circle_number), point, cv.FONT_HERSHEY_SIMPLEX, 0.7, SOLID_YELLOW, 2)
    return cv.circle(
        out_frame,
        point,
        10,
        (192, 133, 156),
        2
    )


class BirdViewWriter:
    def __init__(self, output_path, fps, shape):
        self.output_path = output_path
        self.fps = fps
        self.frame_width = shape[0]
        self.frame_height = shape[1]
        self.blank_image = np.zeros((self.frame_height, self.frame_width, 3), np.uint8)
        self.blank_image[:] = SOLID_BLACK_COLOR
        self.current_frame = 1
        self.writer = OutputVideoWriter(self.output_path, self.fps, shape)

    def __del__(self):
        self.writer.release()

    def digest(self, person_boxes, draw_polygon=False):
        out_frame = np.copy(self.blank_image)
        if len(person_boxes) > 0:
            person_boxes_ordered = self.order_boxes(person_boxes)
            print(f'Digesting a total of {len(person_boxes)} person detections')
            for i, person_box in enumerate(person_boxes_ordered):
                x, y = self.middle_of_box(person_box)
                out_frame = draw_circle(out_frame, i, (x, y))

            if draw_polygon:
                self.draw_polygon(out_frame, person_boxes_ordered)
        self.writer.write(out_frame)
        self.current_frame += 1

    def draw_polygon(self, out_frame, person_boxes):
        if len(person_boxes) != 2:
            print(f'Warning: length of person_boxes list is not 2 (it is {len(person_boxes)}). \n'
                  'Therefore it is impossible to draw the polygon...')
            return out_frame
        edges = [self.middle_of_box(person_boxes[0]), self.middle_of_box(person_boxes[1])]

        # move first box to the right:
        box_height = person_boxes[0][2] - person_boxes[0][0]
        third_edge = [person_boxes[0][0], person_boxes[0][1] + box_height, person_boxes[0][2],
                      person_boxes[0][3] + box_height]

        edges.append(self.middle_of_box(third_edge))

        # draw polygon:
        edges = np.array(edges, np.int32)
        edge_pts = edges.reshape((-1, 1, 2))
        return cv.polylines(out_frame, [edge_pts], True, color=(0, 0, 255), thickness=10)

    def release(self):
        self.__del__()

    def middle_of_box(self, box):
        x_mid = (box[1] * self.frame_width + box[3] * self.frame_width) / 2
        y_mid = (box[0] * self.frame_height + box[2] * self.frame_height) / 2
        return int(x_mid), int(y_mid)

    def order_boxes(self, person_boxes):
        person_boxes = person_boxes[person_boxes[:, 2].argsort()]
        person_boxes = person_boxes[person_boxes[:, 1].argsort(kind='merge')]
        return person_boxes
