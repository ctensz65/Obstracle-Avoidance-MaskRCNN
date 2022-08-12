from glob import glob
import logging
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
        # model: This MUST be a tflite model that was specifically compiled for Edge TPU.
        # https://coral.withgoogle.com/web-compiler/
        # logging.info('Creating a ObjectsOnRoadProcessor...')
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
        self.traffic_objects = {5: GreenTrafficLight(),
                                1: Bola(140),
                                0: Kerucut(),
                                2: Sarden()}

        self.arahJalan = {0: "mundur",
                          1: "lurus",
                          2: "kanan",
                          3: "kiri"}

    def get_boxes(self, outputs):
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

        return pred_boxes

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
        # instances_label = instances_label.tolist()

        return instances_label

    def process_objects_on_road(self, frame, prediction):
        # Main entry point of the Road Object Handler
        logging.debug('Processing objects.................................')
        global topleft, bottomright

        if frame is not None:
            boxes = self.get_boxes(prediction)
            label = self.get_labels(prediction)
        else:
            logging.debug(
                'There Is no Frame to Process.................................')

        if label.nelement() != 0:
            self.heightObj, self.widthObj = self.get_hwframe(prediction)
            topleft, bottomright = self.get_coords(prediction)

        self.control_car(boxes, label, self.heightObj)
        logging.debug('Processing objects END..............................')

        arah = self.arahJalan[self.flagBelok]
        # print ("Speed Motor : ", self.speed)
        return self.speed, arah

    def control_car(self,
                    boxes,
                    label,
                    height
                    ):
        global topleft, bottomright

        logging.debug('Control car...')
        car_state = {"speed": self.speed_limit,
                     "speed_limit": self.speed_limit}

        if len(boxes) == 0:
            self.flagBelok = 1
            logging.debug(
                'No objects detected, drive at speed limit of %s.' % self.speed_limit)

        contain_stop_sign = False

        if label.nelement() != 0:
            # print ("OK")
            label = int(label[0])
            processor = self.traffic_objects[label]
            if processor.is_close_by(height, self.height):
                processor.set_car_state(car_state)
                self.flagBelok = processor.check_lebar(
                    topleft, bottomright, self.width)
            else:
                logging.debug(
                    "[%s] object detected, but it is too far, ignoring. " % label)
            if label == '2':
                contain_stop_sign = True

        if not contain_stop_sign:
            self.traffic_objects[2].clear()

        self.resume_driving(car_state)

    def resume_driving(self, car_state):
        old_speed = self.speed
        self.speed_limit = car_state['speed_limit']
        self.speed = car_state['speed']

        logging.debug('Current Speed = %d, New Speed = %d' %
                      (old_speed, self.speed))

        if self.speed == 0:
            logging.debug('full stop for 1 seconds')
            time.sleep(1)
