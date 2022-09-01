import cv2
import torch
from collections import deque
import time

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
    def __init__(self, instance_mode=ColorMode.IMAGE, parallel=False, height=480, speedlimit=100):
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

        # self.cfg = cfg
        self.processing = ObjectsOnRoadProcessor(self, speed_limit=speedlimit)

        self.parallel = parallel
        # self.predictor = DefaultPredictor(self.cfg)

        self.new_frame_time = 0
        self.prev_frame_time = 0

        self.speed = 0
        self.arahJalan = ""
        self.fps = 0

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
            flagtrain = True

        if dataset_val in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_val)

        for d in ["train", "val"]:
            register_coco_instances(
                f"Components_{d}", {}, f"{pathdataset}/{d}.json", f"{pathdataset}/{d}")

        return flagtrain

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

            if "instances" in predictions:
                predictions1 = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(
                    frame1, predictions1)

                label_obj = predictions['instances'].pred_classes.to(
                    "cpu").numpy()

            self.fps = 1/(self.new_frame_time-self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time

            # converting the fps into integer
            self.fps = int(self.fps)

            # converting the fps to string so that we can display it on frame
            # by using putText
            string_fps = (str(self.fps) + " fps")

            try:
                int_object = label_obj[0]
                if int_object == 0:
                    self.object = 'Bola Kasti'
                elif int_object == 1:
                    self.object = 'Kerucut'
                elif int_object == 2:
                    self.object = 'Sarden'
            except IndexError:
                self.object = 'No Object Detected'

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

            # Send to Arduino
            # toArduino = ("<" + self.arahJalan + ", " + str(self.speed) + ">")
            # print()
            # print(toArduino)
            # write_data(toArduino)

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
        return self.fps, self.speed, self.arahJalan, self.object
