from tkinter import Y
import cv2
import torch
from collections import deque
import time
import numpy as np

from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


from processing_object import ObjectsOnRoadProcessor

pathdataset = "..\..\dataset"
model_path = "model_final_ske2.pth"
config_path = "..\..\configs\COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


class VisualizationDemo(object):
    def __init__(self, instance_mode=ColorMode.IMAGE, parallel=False, height=480, speedlimit=150):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get("Components_train")
        # self.datasetdict = DatasetCatalog.get("Components_train")
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        # self.cfg = cfg
        self.processing = ObjectsOnRoadProcessor(self, speed_limit=speedlimit)

        self.parallel = parallel
        # self.predictor = DefaultPredictor(self.cfg)

        self.new_frame_time = 0
        self.prev_frame_time = 0
        self.time_elapsed = 0
        self.secoundCounter = 0

        self.speed = 0
        self.arahJalan = ""
        self.fps = 0

        self.scoreInt = 0

    def get_model(self, nobject, threshold):
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.WEIGHTS = model_path
        cfg.DATASETS.TEST = ("Components_val", )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = nobject
        cfg.freeze()

        return cfg

    def get_predict(self, nobject, threshold):
        cfg = self.get_model(nobject, threshold)

        self.predictor = DefaultPredictor(cfg)

    def register_dataset(self):
        dataset_train = 'Components_train'
        dataset_val = 'Components_val'

        if dataset_train in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_train)

        if dataset_val in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_val)

        for d in ["train", "val"]:
            register_coco_instances(
                f"Components_{d}", {}, f"{pathdataset}/{d}.json", f"{pathdataset}/{d}")

        datasetdict = DatasetCatalog.get("Components_train")
        return datasetdict, self.metadata

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def validasi_dataset(self, frame):
        img = cv2.imread(frame["file_name"])
        image = img[:, :, ::-1]

        visualizer = Visualizer(image, self.metadata, scale=0.75)
        vis_output = visualizer.draw_dataset_dict(frame)

        return vis_output

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.new_frame_time = time.time()

            self.secoundCounter += 1
            time.gmtime(0)
            self.time_elapsed = time.strftime(
                "%H:%M:%S", time.gmtime(self.secoundCounter))

            # Initiate Control Car
            self.speed, self.arahJalan = self.processing.process_objects_on_road(
                frame1, predictions)

            if "instances" in predictions:
                predictions1 = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(
                    frame1, predictions1)

                label_obj = predictions['instances'].pred_classes.to(
                    "cpu").numpy()

                pred_boxes = predictions['instances'].pred_boxes.tensor.cpu(
                ).numpy()

                pred_scores = predictions['instances'].scores.to("cpu").numpy()

            self.fps = 1/(self.new_frame_time-self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time

            # converting the fps into integer
            self.fps = int(self.fps)

            try:
                self.Objwidth = pred_boxes[0][2] - pred_boxes[0][0]
                self.Objheight = pred_boxes[0][3] - pred_boxes[0][1]

                self.Objwidth = float("{:.2f}".format(self.Objwidth))
                self.Objheight = float("{:.2f}".format(self.Objheight))

                pred_scores.astype(int)
                self.score = '{:.0%}'.format(pred_scores[0] / 1)
                self.scoreInt = pred_scores

                int_object = label_obj[0]
                if int_object == 0:
                    self.object = 'Bola Kasti'
                elif int_object == 1:
                    self.object = 'Kerucut'
                elif int_object == 2:
                    self.object = 'Sarden'

                self.center_coor = self.compute_center(pred_boxes)
                self.center_coor.astype(int)

                x = "{:.1f}".format(self.center_coor[0][0])
                y = "{:.1f}".format(self.center_coor[0][1])
                z = "{:.1f}".format(self.center_coor[0][2])

                self.center_coor = ("X:" + str(x) + " Y:" +
                                    str(y) + " Z:" + str(z))

            except IndexError:
                self.object = 'No Object Detected'
                self.center_coor = 'None'
                self.Objwidth = 'None'
                self.Objheight = 'None'
                self.score = 0

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))

    def data_atribut(self):
        return self.fps, self.speed, self.arahJalan, self.object, self.Objheight, self.Objwidth, self.center_coor, self.score, self.time_elapsed, self.scoreInt

    def compute_center(self, bounding_boxes):
        # type: (np.ndarray) -> np.ndarray
        """
        Computes the bounding box centers given the bounding boxes.
        Args:
            bounding_boxes: numpy array containing two diagonal corner coordinates of the bounding boxes,
            shape [num_bounding_boxes, 4]
        Returns:
            centers_3d: numpy array of the bounding box centers, shape [num_bounding_boxes, 3]
        """
        x_dist = bounding_boxes[:, 2] - bounding_boxes[:, 0]
        y_dist = bounding_boxes[:, 3] - bounding_boxes[:, 1]
        centers = bounding_boxes[:, 0:2] + 0.5 * \
            np.stack((x_dist, y_dist), axis=1)
        centers_3d = self.add_z_coordinate(centers)
        return centers_3d

    def add_z_coordinate(self, centers):
        # type: (np.ndarray) -> np.ndarray
        """
        Adds a dummy 0. z-coordinate to the object centers.
        Could be replaced with some depth estimation algorithm in the future.
        Args:
            centers: 2d coordinates for the observed bounding box centers, shape [num_centers, 2]
        :return:
            3d coordinates for the observed bounding box centers, shape [num_centers, 2]
        """
        return np.concatenate((centers, np.zeros(shape=(centers.shape[0], 1))), axis=1)
