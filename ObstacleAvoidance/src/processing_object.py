from glob import glob
import time
from tkinter import E
from objects import *

from detectron2.config import get_cfg

############################
# Frame processing steps
############################

topleft = 0
bottomright = 0


class ObjectsOnRoadProcessor(object):
    """
    This class 1) detects what objects (namely traffic signs and people) are on the road
    and 2) controls the car navigation (speed/steering) accordingly
    """

    def __init__(self,
                 car=None,
                 speed_limit=100,
                 width=640,
                 height=480):
        self.width = width
        self.height = height
        self.arahJalan = ""
        self.flagBelok = 1

        self.heightObj = 0
        self.widthObj = 0

        # initialize car
        self.car = car
        self.speed_limit = speed_limit
        self.speed = speed_limit

        # detectron2 cfg
        self.predictor = None

        #
        '''
        Mapping Penomoran Label
        0 = Bola Kasti
        1 = Kerucut
        2 = Sarden
        '''
        self.traffic_objects = {1: Kerucut(255),
                                0: Bola(),
                                2: Sarden(255),
                                3: Kacamata()}

        self.arahJalan = {0: "stop",
                          1: "lurus",
                          2: "kanan",
                          3: "kiri",
                          4: "mundur",
                          5: "mundur_kiri",
                          6: "mundur_kanan"}

    def get_boxes(self, outputs):
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

        return pred_boxes

    def get_center(self, outputs):
        instances_pred_boxes = outputs["instances"].pred_boxes.tensor.cpu(
        ).numpy()

        x_dist = instances_pred_boxes[:, 2] - instances_pred_boxes[:, 0]
        y_dist = instances_pred_boxes[:, 3] - instances_pred_boxes[:, 1]
        centers = instances_pred_boxes[:, 0:2] + \
            0.5 * np.stack((x_dist, y_dist), axis=1)

        return centers

    def get_coords(self, outputs):
        """
        Untuk dari top left dan bottom right
        """
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

        coord_top_left = int(pred_boxes[0][0])
        coord_bottom_right = int(pred_boxes[0][2])

        return coord_top_left, coord_bottom_right

    def get_hwframe(self, outputs):
        """
        Untuk mengambil height dan width dari object.
        """
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        width = pred_boxes[0][2] - pred_boxes[0][0]
        height = pred_boxes[0][3] - pred_boxes[0][1]

        return height, width

    def get_labels(self, outputs):
        """
        Untuk mengambil height dan width dari object. label == classes_id
        [0] Bola
        [1] Kerucut
        [2] Sarden
        """
        instances_label = outputs['instances'].pred_classes
        label_list = outputs['instances'].pred_classes.to("cpu").numpy()
        # instances_label = instances_label.tolist()

        return instances_label, label_list

    def process_objects_on_road(self, frame, prediction):
        # Main entry point of the Road Object Handler
        global topleft, bottomright, center_coord

        if frame is not None:
            boxes = self.get_boxes(prediction)
            label, label_list = self.get_labels(prediction)

        if label.nelement() != 0:
            self.heightObj, self.widthObj = self.get_hwframe(prediction)
            topleft, bottomright = self.get_coords(prediction)
            center_coord = self.get_center(prediction)

        self.control_car(boxes, label, self.heightObj)

        arah = self.arahJalan[self.flagBelok]
        # print ("Speed Motor : ", self.speed)
        return self.speed, arah

    def control_car(self,
                    boxes,
                    label,
                    height
                    ):
        global topleft, bottomright, centercoord

        car_state = {"speed": self.speed_limit,
                     "speed_limit": self.speed_limit}

        if len(boxes) == 0:
            self.flagBelok = 1

        contain_stop_sign = False

        if label.nelement() != 0:
            # print ("OK")
            label = int(label[0])
            processor = self.traffic_objects[label]
            if processor.is_close_by(height, self.height):
                processor.set_car_state(car_state)
                # jika bola kasti maka stop
                if label == 0:
                    self.flagBelok = 0
                elif label == 2:
                    self.flagBelok = processor.check_mundur(
                        centercoord, self.width)
                else:
                    self.flagBelok = processor.check_lebar(
                        topleft, bottomright, self.width)

            if label == '3':
                contain_stop_sign = True

        if not contain_stop_sign:
            self.traffic_objects[3].clear()

        self.resume_driving(car_state)

    def resume_driving(self, car_state):
        old_speed = self.speed
        self.speed_limit = car_state['speed_limit']
        self.speed = car_state['speed']

        if self.speed == 0:
            time.sleep(1)
