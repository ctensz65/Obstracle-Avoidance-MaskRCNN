from cProfile import label
from telnetlib import X3PAD
import streamlit as st
# import time
import tqdm
import cv2
import tempfile
import os
import numpy as np
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from detector2 import VisualizationDemo
from SendToArduino import *

pathdataset = "..\..\dataset"
model_path = "model_final_ske2.pth"
config_path = "..\..\configs\COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


def get_model(nobject, threshold):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_path
    cfg.DATASETS.TEST = ("Components_val", )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nobject
    cfg.freeze()
    return cfg


def register_dataset():
    for d in ["train", "val"]:
        register_coco_instances(
            f"Components_{d}", {}, f"{pathdataset}/{d}.json", f"{pathdataset}/{d}")


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def welcome():
    st.markdown('App Made using **Detectron2** & **Pytorch**')

    st.markdown(
        """
        <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
    width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
    width: 350px
    margin-left: -350px
    </style>
        """, unsafe_allow_html=True,)


def webcam():
    DatasetCatalog.clear()
    register_dataset()
    test_metadata = MetadataCatalog.get("Components_train")

    st.sidebar.markdown('---------')
    st.markdown(
        """
        <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
    width: 450px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
    width: 450px
    margin-left: -350px
    </style>
        """, unsafe_allow_html=True,)

    st.header("Webcam Live Feed")
    columns = st.columns((2, 1, 2))
    with columns[2]:
        stframe = st.empty()
    st.markdown("<hr/>", unsafe_allow_html=True)

    # tfile = tempfile.NamedTemporaryFile(delete=False)

    detection_confidence = st.sidebar.slider(
        'Confidence Threshold', min_value=0.0, max_value=1.0, value=0.5)

    jumlah_objek = st.sidebar.number_input(
        'Jumlah Objek', min_value=1, max_value=5, value=3)

    kecepatan_awal = st.sidebar.slider(
        'Kecepatan Konstan', min_value=80, max_value=110, value=100)

    start_button = st.sidebar.button("Start Engine")
    stop_button = st.sidebar.button("Stop Video")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Frame Rate</strong></p>'
        initial_value = '<p style="text-align: center;">0</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi1_text = st.markdown(initial_value, unsafe_allow_html=True)
    with kpi2:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Arah Rover</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi2_text = st.markdown(initial_value, unsafe_allow_html=True)
    with kpi3:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Kecepatan</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi3_text = st.markdown(initial_value, unsafe_allow_html=True)
    with kpi4:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Camera Width</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi4_text = st.markdown(initial_value, unsafe_allow_html=True)
    with kpi5:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Label</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi5_text = st.markdown(initial_value, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    x1, y1, x2, y2, score = st.columns(5)
    with x1:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>X1</strong></p>'
        initial_value = '<p style="text-align: center;">0</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        x1_text = st.markdown(initial_value, unsafe_allow_html=True)
    with y1:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Y1</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        y1_text = st.markdown(initial_value, unsafe_allow_html=True)
    with x2:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>X2</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        x2_text = st.markdown(initial_value, unsafe_allow_html=True)
    with y2:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Y2</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        y1_text = st.markdown(initial_value, unsafe_allow_html=True)
    with score:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Score Prediction</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        score_text = st.markdown(initial_value, unsafe_allow_html=True)

    if start_button:
        cam_warning = st.warning("Camera loading, hold on...")
        cam = cv2.VideoCapture(0)
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        width = (str(width) + " px")
        cam_warning.empty()

        weight_warning = st.warning(
            "Loading model, might take a few minutes, hold on...")
        cfg = get_model(jumlah_objek, detection_confidence)
        kecepatan_awal = int(kecepatan_awal)
        demo = VisualizationDemo(cfg, speedlimit=kecepatan_awal)
        weight_warning.empty()

        for vis in tqdm.tqdm(demo.run_on_video(cam, test_metadata)):
            # columns = st.columns((2, 1, 2))
            # with columns[1]:
            #     stframe.image(vis, channels='BGR',
            #                   use_column_width=False, caption="Running model")

            stframe.image(vis, channels='BGR', use_column_width=False)

            fps, speed, arahJalan, label = demo.data_atribut()

            kpi1_text.write(
                f"<h3 style='text-align: center; color:yellow; '>{fps}</h3>", unsafe_allow_html=True)

            kpi2_text.write(
                f"<h3 style='text-align: center; color:yellow; '>{arahJalan}</h3>", unsafe_allow_html=True)

            kpi3_text.write(
                f"<h3 style='text-align: center; color:yellow; '>{speed}</h3>", unsafe_allow_html=True)

            kpi4_text.write(
                f"<h3 style='text-align: center; color:yellow; '>{width}</h3>", unsafe_allow_html=True)

            kpi5_text.write(
                f"<h3 style='text-align: center; color:yellow; '>{label}</h3>", unsafe_allow_html=True)

            if stop_button:
                start_button = False
                closed_connection()
                break
        closed_connection()
        cam.release()

        '''
        while cam.isOpened():
            ret, image = cam.read()
            if not ret:
                continue

            new_frame_time = time.time()

            output = predictor(image)

            vis = Visualizer(image[:, :, ::-1],
                             metadata=test_metadata, scale=1.2)
            vis = vis.draw_instance_predictions(output["instances"].to("cpu"))

            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            fps = int(fps)
            kpi1_text.write(
                f"<h1 style='text-align: center; color:red; '>{fps}</h1>", unsafe_allow_html=True)

            stframe.image(vis.get_image()[:, :, ::-1],
                          channels='BGR', use_column_width=False)

            kpi3_text.write(
                f"<h1 style='text-align: center; color:red; '>{width}</h1>", unsafe_allow_html=True)

            if stop_button:
                start_button = False
                break
        cam.release()
        '''


def main():
    st.set_page_config(
        page_title="RoverBot",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title('Obstracle Avaoidance using Image Segmentation Mask RCNN')

    st.markdown(
        """
        <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
    width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
    width: 350px
    margin-left: -350px
    </style>
        """, unsafe_allow_html=True,)

    # DatasetCatalog.clear()
    # dataset_name = ['Components_train', 'Component_val']
    # if dataset_name in DatasetCatalog.list():
    #     DatasetCatalog.remove(dataset_name)
    # register_dataset()

    mode = ['Welcome', 'Rover Web Monitoring']
    app_mode = st.sidebar.selectbox(
        'Pilih Halaman', mode)
    if app_mode == 'Welcome':
        welcome()
    elif app_mode == 'Rover Web Monitoring':
        webcam()


if __name__ == "__main__":
    main()
