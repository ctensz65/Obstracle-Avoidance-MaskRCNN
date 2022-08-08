from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import cv2
import argparse

WINDOW_NAME = 'Bisa Yuk'
DatasetFold = "..\dataset"

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
        register_coco_instances(f"Components_{d}", {}, f"{DatasetFold}/{d}.json", f"{DatasetFold}/{d}")

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

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, image = cap.read()
            
        outputs = predictor(image)

        v = Visualizer(image[:, :, ::-1], metadata=test_metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow(WINDOW_NAME, v.get_image()[:, :, ::-1])
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cap.release()
    cv2.destroyAllWindows()
    