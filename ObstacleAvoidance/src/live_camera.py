from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import cv2
import torch
import argparse
import matplotlib as mpl

from processing_object import ObjectsOnRoadProcessor
from SendToArduino import Velocity

WINDOW_NAME = 'Bisa Yuk'

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

        self.predictor = DefaultPredictor(cfg)
        self.height = height
        self.width = 640

        # initialize open cv for drawing boxes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, height - 10)
        self.fontScale = .8
        self.fontColor = (255, 255, 255)  # white
        self.lineType = 2
        
    def process_predictions(self, frame, predictions):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initiate Control Car
        speed = self.processing.process_objects_on_road(frame, predictions)
        
        # Text
        annotate_summary = ("Velocity : " + str(speed))

        v = Visualizer(frame[:, :, ::-1], metadata=test_metadata, scale=1.2)
        vis_frame = v.draw_instance_predictions(predictions["instances"].to("cpu"))

        # x1, y1 = pred_boxes[0][1], pred_boxes[0][2]
        text_pos = (15, self.height - 10)
        horiz_align = "right"
        vis_frame = v.draw_text(annotate_summary, text_pos, horizontal_alignment=horiz_align)

        # Converts Matplotlib RGB format to OpenCV BGR format
        # vis_frame = cv2.putText(vis_frame, annotate_summary, self.bottomLeftCornerOfText, self.font, self.fontScale, self.fontColor, self.lineType)

        # Send to Arduino
        self.Toarduino.write_data(speed)
        
        return vis_frame

def get_model(model_path, config_path, threshold, object):
    # Create config
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_path
    cfg.DATASETS.TEST = ("Components_val", )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = object # Jumlah classes

    return DefaultPredictor(cfg), cfg

def register_dataset():
    for d in ["train", "val"]:      
        register_coco_instances(f"Components_{d}", {}, f"/home/falco/codes/Skenario2/{d}.json", f"/home/falco/codes/Skenario2/{d}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects from webcam images')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config')
    parser.add_argument('-t', '--threshold', type=int, default=0.5, help='Detection threshold')
    parser.add_argument('-o', '--object', type=int, default=3, help='Berapa Banyak Object?')
    args = parser.parse_args()

    register_dataset()
    predictor, cfg = get_model(args.model, args.config, args.threshold, args.object)
    
    dataset_dicts = DatasetCatalog.get("Components_train")
    test_metadata = MetadataCatalog.get("Components_train")

    cap = cv2.VideoCapture('http://Cartensz-PC.local:8000/camera/mjpeg?type=.mjpg')
    demo = VisualizationDemo(cfg)

    while cap.isOpened():
        ret, image = cap.read()

        output = predictor(image)
        
        vis = demo.process_predictions(image, output)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, vis.get_image()[:, :, ::-1])
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cap.release()
    cv2.destroyAllWindows()
    