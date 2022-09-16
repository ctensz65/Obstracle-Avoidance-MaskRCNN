import cv2
import torch
from collections import deque
import time

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from processing_object import ObjectsOnRoadProcessor
from SendToArduino import write_data


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False, height=480, speedlimit=100):
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
        self.processing = ObjectsOnRoadProcessor(self, speed_limit=speedlimit)

        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)

        self.new_frame_time = 0
        self.prev_frame_time = 0

        self.speed = 0
        self.arahJalan = ""
        self.fps = 0

        # initialize open cv for drawing boxes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, height - 10)
        self.upperLeftCornerOfText = (10, 15)
        self.fontScale = .5
        self.fontColor = (255, 255, 255)  # white
        self.lineType = 1

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

            # Initiate Control Car
            self.speed, self.arahJalan = self.processing.process_objects_on_road(
                frame1, predictions)

            # Text
            annotate_summary = ("Setpoint: " + str(self.speed))

            if "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(
                    frame1, predictions)

            self.fps = 1/(self.new_frame_time-self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time

            # converting the fps into integer
            self.fps = int(self.fps)

            # converting the fps to string so that we can display it on frame
            # by using putText
            string_fps = (str(self.fps) + " fps")

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            vis_frame = cv2.putText(vis_frame, annotate_summary, self.bottomLeftCornerOfText,
                                    self.font, self.fontScale, self.fontColor, self.lineType)
            vis_frame = cv2.putText(
                vis_frame, string_fps, self.upperLeftCornerOfText, self.font,
                self.fontScale, self.fontColor, self.lineType)

            # Send to Arduino
            toArduino = ("<" + self.arahJalan + ", " + str(self.speed) + ">")
            # print()
            # print(toArduino)
            write_data(toArduino)

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
        return self.fps, self.speed, self.arahJalan
