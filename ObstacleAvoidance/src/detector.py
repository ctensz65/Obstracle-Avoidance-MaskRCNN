from concurrent.futures import process
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import multiprocessing as mp

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg

from detectron2.data.datasets import register_coco_instances
from processing_object import ObjectsOnRoadProcessor
from SendToArduino import Velocity

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False, height=480):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get("Components_train")
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        
        self.cfg = cfg
        self.processing = ObjectsOnRoadProcessor(self)
        self.Toarduino = Velocity()

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            # self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

        # initialize open cv for drawing boxes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, height - 10)
        self.fontScale = .8
        self.fontColor = (255, 255, 255)  # white
        self.lineType = 2

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

    def get_rawpredict(self, video):
        """
        Untuk mengambil raw prediction
        """
        frame = self._frame_from_video(video)
        outputs = self.predictor(frame)

        return outputs

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]

        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

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

            # Initiate Control Car
            speed = self.processing.process_objects_on_road(frame1, predictions)

            # Text
            annotate_summary = ("Velocity : " + str(speed))
            
            if "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame1, predictions)
            
            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            vis_frame = cv2.putText(vis_frame, annotate_summary, self.bottomLeftCornerOfText, self.font, self.fontScale, self.fontColor, self.lineType)

            # Send to Arduino
            self.Toarduino.write_data(speed)
            
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

