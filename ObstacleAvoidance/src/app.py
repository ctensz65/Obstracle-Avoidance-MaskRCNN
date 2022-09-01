import streamlit as st
import time
import tqdm
import cv2
import tempfile
import os
import numpy as np
import serial

# from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from detector2 import VisualizationDemo

pathdataset = "..\..\dataset"
model_path = "model_final_ske2.pth"
config_path = "..\..\configs\COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

startMarker = 60
endMarker = 62
listeningMode = False


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


def initialize_atribut():
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

    return kpi1_text, kpi2_text, kpi3_text, kpi4_text, kpi5_text, x1_text, y1_text, x2_text, y1_text, score_text


def webcam():
    global listeningMode

    # def sendToArduino(sendStr):
    #     arduino.write(sendStr.encode('utf-8'))  # change for Python3

    # def recvFromArduino():
    #     global startMarker, endMarker

    #     ck = ""
    #     x = "z"  # any value that is not an end- or startMarker
    #     byteCount = -1  # to allow for the fact that the last increment will be one too many

    #     # wait for the start character
    #     while ord(x) != startMarker:
    #         x = arduino.read()

    #     # save data until the end marker is found
    #     while ord(x) != endMarker:
    #         if ord(x) != startMarker:
    #             ck = ck + x.decode("utf-8")  # change for Python3
    #             byteCount += 1
    #         x = arduino.read()

    #     return(ck)

    # def write_data(value):
    #     waitingForReply = False
    #     if arduino.isOpen():
    #         if waitingForReply == False:
    #             sendToArduino(value)
    #             print("\nSent from PC " + value)
    #             waitingForReply = True

    #         if waitingForReply == True:
    #             while arduino.inWaiting() == 0:
    #                 pass

    #             dataRecvd = recvFromArduino()
    #             print("Reply Received " + dataRecvd)
    #             waitingForReply = False

    #             print("===========")

    #         time.sleep(0.1)
    #     else:
    #         print("Arduino Port Not Open !")
    #         arduino.close()

    # def closed_connection():
    #     msgStop = "<stop, 0>"
    #     sendToArduino(msgStop)
    #     arduino.flushInput()
    #     arduino.flushOutput()
    #     arduino.close()
    #     print("GoodBye")

    # dataset_name = ['Components_train', 'Components_val']

    # if dataset_name in DatasetCatalog.list():
    #     DatasetCatalog.remove(dataset_name)
    # DatasetCatalog.clear()
    # register_dataset()
    # test_metadata = MetadataCatalog.get("Components_train")

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
    with columns[1]:
        stframe = st.empty()
    st.markdown("<hr/>", unsafe_allow_html=True)

    detection_confidence = st.sidebar.slider(
        'Confidence Threshold', min_value=0.0, max_value=1.0, value=0.5)

    jumlah_objek = st.sidebar.number_input(
        'Jumlah Objek', min_value=1, max_value=5, value=3)

    kecepatan_awal = st.sidebar.slider(
        'Kecepatan Konstan', min_value=80, max_value=110, value=100)

    main_button = st.sidebar.empty()
    engine_button = st.sidebar.empty()

    kpi1_text, kpi2_text, kpi3_text, kpi4_text, kpi5_text, x1_text, y1_text, x2_text, y1_text, score_text = initialize_atribut()

    if 'button' not in st.session_state:
        st.session_state.button = False

    if 'close' not in st.session_state:
        st.session_state.close = False

    # arduino_warning = st.warning(
    #     "Initialize Arduino Communication, hold on...")
    # arduino = serial.Serial(
    #     'COM8', baudrate=115200, timeout=.1)
    # time.sleep(2)
    # arduino.flushInput()
    # arduino.flushOutput()
    # arduino.flush()
    # arduino_warning.empty()

    if main_button.button('Start Camera', key='start'):
        cam_warning = st.warning("Camera loading, hold on...")
        cam = cv2.VideoCapture(0)
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        width = (str(width) + " px")
        cam_warning.empty()

        weight_warning = st.warning(
            "Loading model, might take a few minutes, hold on...")

        kecepatan_awal = int(kecepatan_awal)
        demo = VisualizationDemo(speedlimit=kecepatan_awal)
        flagtrain = demo.register_dataset()
        demo.get_predict(jumlah_objek, detection_confidence)

        kecepatan_awal = int(kecepatan_awal)
        weight_warning.empty()

        if flagtrain:
            st.sidebar.write(flagtrain)

        if main_button.button('Stop Camera', key='stop'):
            st.session_state.close = True

        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            stframe.image(vis, channels='BGR', use_column_width=False)

            fps, speed, arahJalan, label = demo.data_atribut()

            # toArduino = ("<" + arahJalan + ", " + str(speed) + ">")
            # write_data(toArduino)
            x1_text.write(
                f"<h3 style='text-align: center; color:yellow; '>Receiving Mode</h3>", unsafe_allow_html=True)

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

            if st.session_state.close:
                # closed_connection()
                main_button.empty()
                engine_button.empty()
                break

        st.write(st.session_state.close)
        cam.release()


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

    mode = ['Welcome', 'Rover Web Monitoring']
    app_mode = st.sidebar.selectbox(
        'Pilih Halaman', mode)
    if app_mode == 'Welcome':
        welcome()
    elif app_mode == 'Rover Web Monitoring':
        webcam()


if __name__ == "__main__":
    main()
