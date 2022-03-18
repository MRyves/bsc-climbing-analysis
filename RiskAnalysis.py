from writer.BirdViewWriter import BirdViewWriter, BirdViewWriter2


class RiskAnalysis:
    def __init__(self, frame_shape, output_writer=None):
        self.output_writer = None
        self.frame_shape = frame_shape
        if output_writer is not None:
            self.output_writer = BirdViewWriter2(output_writer, self.frame_shape)

    def analyze(self, person_boxes):
        print(f'Risk analyzing a total of {len(person_boxes)} detection boxes...')
        person_boxes = self.__order_boxes(person_boxes)
        box_centers = [self.__middle_of_box(box) for box in person_boxes]
        self.write(box_centers, [])

    def finished_analysis(self):
        if self.has_output_writer:
            self.output_writer.release()

    def write(self, circles, polygon_edges):
        if self.has_output_writer:
            self.output_writer.write(circles, polygon_edges)

    def __order_boxes(self, person_boxes):
        person_boxes = person_boxes[person_boxes[:, 2].argsort()]
        person_boxes = person_boxes[person_boxes[:, 1].argsort(kind='merge')]
        return person_boxes

    def __middle_of_box(self, box):
        x_mid = (box[1] * self.frame_width + box[3] * self.frame_width) / 2
        y_mid = (box[0] * self.frame_height + box[2] * self.frame_height) / 2
        return int(x_mid), int(y_mid)

    @property
    def has_output_writer(self):
        return self.output_writer is not None

    @property
    def frame_width(self):
        return self.frame_shape[0]

    @property
    def frame_height(self):
        return self.frame_shape[1]
