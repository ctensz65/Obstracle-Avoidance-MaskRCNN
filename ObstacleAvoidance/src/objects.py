from threading import Timer
import logging
import numpy as np


class TrafficObject(object):
    def set_car_state(self, car_state):
        pass

    @staticmethod
    def is_close_by(objHeight, frame_height, min_height_pct=0.1):
        # default: if a sign is 10% of the height of frame
        # obj_height = obj.bounding_box[1][1]-obj.bounding_box[0][1]
        float_formatter = "{:.2f}".format
        np.set_printoptions(formatter={'float_kind': float_formatter})
        objHeight = np.array(objHeight)

        asu = objHeight / frame_height > min_height_pct
        return asu

    @staticmethod
    def check_lebar(objXleft, objXright, frame_width):
        # default: if a sign is 10% of the height of frame
        # obj_height = obj.bounding_box[1][1]-obj.bounding_box[0][1]
        float_formatter = "{:.2f}".format
        np.set_printoptions(formatter={'float_kind': float_formatter})
        objXleft = np.array(objXleft)
        objXright = np.array(objXright)

        distanceLeft = abs(0 - objXleft)
        distanceRight = frame_width - objXright

        # print(distanceLeft)
        # print(distanceRight)

        if distanceLeft > distanceRight:
            arahJalan = 3
            return arahJalan
        elif distanceLeft < distanceRight:
            arahJalan = 2
            return arahJalan
        else:
            arahJalan = 1
            return arahJalan

        # print(frame_width)


class GreenTrafficLight(TrafficObject):

    def set_car_state(self, car_state):
        logging.debug('green light: make no changes')


class Bola(TrafficObject):
    """
    Ekivalen dengan SpeedLimit to 25
    """

    def __init__(self, speed_limit):
        self.speed_limit = speed_limit

    def set_car_state(self, car_state):
        car_state['speed'] = self.speed_limit
        # return (car_state)


class Kerucut(TrafficObject):
    """
    Ekivalen dengan RedLightTraffic
    """

    def set_car_state(self, car_state):
        logging.debug('red light: stopping car')
        car_state['speed'] = 0


class Sarden(TrafficObject):
    """
    Stop Sign object would wait
    """

    def __init__(self, wait_time_in_sec=10, min_no_stop_sign=20):
        self.in_wait_mode = False
        self.has_stopped = False
        self.wait_time_in_sec = wait_time_in_sec
        self.min_no_stop_sign = min_no_stop_sign
        self.no_stop_count = min_no_stop_sign
        self.timer = None

    def set_car_state(self, car_state):
        self.no_stop_count = self.min_no_stop_sign

        if self.in_wait_mode:
            logging.debug('stop sign: 2) still waiting')
            # wait for 2 second before proceeding
            car_state['speed'] = 0
            return

        if not self.has_stopped:
            logging.debug('stop sign: 1) just detected')

            car_state['speed'] = 0
            self.in_wait_mode = True
            self.has_stopped = True
            self.timer = Timer(self.wait_time_in_sec, self.wait_done)
            self.timer.start()
            return

    def wait_done(self):
        logging.debug('stop sign: 3) finished waiting for %d seconds' %
                      self.wait_time_in_sec)
        self.in_wait_mode = False

    def clear(self):
        if self.has_stopped:
            # need this counter in case object detection has a glitch that one frame does not
            # detect stop sign, make sure we see 20 consecutive no stop sign frames (about 1 sec)
            # and then mark has_stopped = False
            self.no_stop_count -= 1
            if self.no_stop_count == 0:
                logging.debug("stop sign: 4) no more stop sign detected")
                self.has_stopped = False
                self.in_wait_mode = False  # may not need to set this
