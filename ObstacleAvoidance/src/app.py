import streamlit as st
import time
import tqdm
import cv2
import tempfile
import os
import numpy as np
import serial
from PIL import Image
import matplotlib.pyplot as plt
import random

# from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from detector2 import VisualizationDemo

pathdataset = "..\..\dataset"
model_path = "model_final_ske2.pth"
config_path = "..\..\configs\COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

listeningMode = False


class ArduinoComm(object):
    def __init__(self, COM='COM8', baudrates=115200):
        self.startMarker = 60
        self.endMarker = 62

        self.arduino = serial.Serial(
            COM, baudrate=baudrates, timeout=.1)
        time.sleep(2)

    def flush_board(self):
        self.arduino.flushInput()
        self.arduino.flushOutput()
        self.arduino.flush()

    def sendToArduino(self, sendStr):
        self.arduino.write(sendStr.encode('utf-8'))  # change for Python3

    def recvFromArduino(self):
        ck = ""
        x = "z"  # any value that is not an end- or startMarker
        byteCount = -1  # to allow for the fact that the last increment will be one too many

        # wait for the start character
        while ord(x) != self.startMarker:
            x = self.arduino.read()

        # save data until the end marker is found
        while ord(x) != self.endMarker:
            if ord(x) != self.startMarker:
                ck = ck + x.decode("utf-8")  # change for Python3
                byteCount += 1
            x = self.arduino.read()

        return(ck)

    def write_data(self, value):
        waitingForReply = False
        if self.arduino.isOpen():
            if waitingForReply == False:
                self.sendToArduino(value)
                print("\nSent from PC " + value)
                waitingForReply = True

            if waitingForReply == True:
                while self.arduino.inWaiting() == 0:
                    pass

                dataRecvd = self.recvFromArduino()
                print("Reply Received " + dataRecvd)
                waitingForReply = False

                print("===========")

            time.sleep(0.1)
        else:
            print("Arduino Port Not Open !")
            self.arduino.close()

    def closed_connection(self):
        msgStop = "<stop, 0>"
        self.sendToArduino(msgStop)
        self.arduino.flushInput()
        self.arduino.flushOutput()
        self.arduino.close()
        print("GoodBye")


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

    st.markdown('OYOYY')

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

    y2, x1, y1, x2, score = st.columns(5)
    with y2:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Status</strong></p>'
        initial_value = '<p style="text-align: center;">0</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        y2_text = st.markdown(initial_value, unsafe_allow_html=True)
    with x1:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Object Height (mm)</strong></p>'
        initial_value = '<p style="text-align: center;">0</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        x1_text = st.markdown(initial_value, unsafe_allow_html=True)
    with y1:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Object Width (mm)</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        y1_text = st.markdown(initial_value, unsafe_allow_html=True)
    with x2:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Center Coordinate</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        x2_text = st.markdown(initial_value, unsafe_allow_html=True)
    with score:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Score Prediction</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        score_text = st.markdown(initial_value, unsafe_allow_html=True)

    return kpi1_text, kpi2_text, kpi3_text, kpi4_text, kpi5_text, x1_text, y1_text, x2_text, y2_text, score_text


def webcam():
    global listeningMode

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

    columns = st.columns((1, 1, 1))
    with columns[1]:
        stframe = st.empty()

    # columns2 = st.columns((2, 1, 2))
    col = st.columns((2.65, 1, 2))
    with col[1]:
        main_button = st.empty()

    st.markdown("<hr/>", unsafe_allow_html=True)

    detection_confidence = st.sidebar.slider(
        'Confidence Threshold', min_value=0.0, max_value=1.0, value=0.5)

    jumlah_objek = st.sidebar.number_input(
        'Jumlah Objek', min_value=1, max_value=5, value=3)

    kecepatan_awal = st.sidebar.slider(
        'Kecepatan Normal', min_value=80, max_value=110, value=100)

    kpi1_text, kpi2_text, kpi3_text, kpi4_text, kpi5_text, x1_text, y1_text, x2_text, y2_text, score_text = initialize_atribut()

    if 'button' not in st.session_state:
        st.session_state.button = False

    if 'close' not in st.session_state:
        st.session_state.close = False

    arduino = ArduinoComm(COM='COM8')

    columns = st.sidebar.columns((1, 1, 1))
    validate = columns[1].button('Validate Dataset', key='validatedataset')
    st.sidebar.markdown('---------')
    if validate:
        kecepatan_awal = int(kecepatan_awal)
        demo = VisualizationDemo(speedlimit=kecepatan_awal)
        dataset_dict, metadata_obj = demo.register_dataset()

        img1, img2, img3 = st.columns(3)

        cntr = 0

        dataset_warn = st.warning(
            "Showing random 3 images to validate dataset ...")
        for d in random.sample(dataset_dict, 3):
            cntr += 1
            visualized_output = demo.validasi_dataset(d)
            fig = plt.figure(figsize=(14, 10))
            plt.imshow(cv2.cvtColor(visualized_output.get_image()
                                    [:, :, ::-1], cv2.COLOR_BGR2RGB))

            if cntr == 1:
                with img1:
                    st.pyplot(fig)
            if cntr == 2:
                with img2:
                    st.pyplot(fig)
            if cntr == 3:
                with img3:
                    st.pyplot(fig)

        objList = metadata_obj.get("thing_classes")
        # st.sidebar.markdown(metadata_obj.get("thing_classes"))

        objCol = st.sidebar.columns((1, 1, 1))
        objCol[1].write(
            f"<h3 style='text-align: center; '>List Objects</h3>", unsafe_allow_html=True)

        objCol2 = st.sidebar.columns((1, 1, 1))
        objCol2[0].caption(
            f"<h3 style='text-align: center; color:yellow; '>{objList[0]}</h3>", unsafe_allow_html=True)

        objCol2[1].caption(
            f"<h3 style='text-align: center; color:yellow; '>{objList[1]}</h3>", unsafe_allow_html=True)

        objCol2[2].caption(
            f"<h3 style='text-align: center; color:yellow; '>{objList[2]}</h3>", unsafe_allow_html=True)

        dataset_warn.empty()

    start_text = '    START ENGINE    '

    if main_button.button(start_text, key='start'):
        cam_warning = st.warning("Starting Camera, hold on .....")
        cam = cv2.VideoCapture(0)
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        width = (str(width) + " px")
        cam_warning.empty()

        weight_warning = st.warning(
            "Loading Mask RCNN Model, hold on .....")
        kecepatan_awal = int(kecepatan_awal)
        demo = VisualizationDemo(speedlimit=kecepatan_awal)
        dataset_dict = demo.register_dataset()
        demo.get_predict(jumlah_objek, detection_confidence)

        kecepatan_awal = int(kecepatan_awal)
        weight_warning.empty()

        arduino.flush_board()

        def store_stop():
            arduino.closed_connection()
            main_button.empty()
            st.success(
                'Engine has been stopped, refresh page to turn on back the engine')

        def write_atributs():
            global speed, arahJalan
            fps, speed, arahJalan, label, Objheight, Objwidth, center_coor, pred_score = demo.data_atribut()

            kpi1_text.write(
                f"<h3 style='text-align: center; color:yellow; '>{fps}</h3>", unsafe_allow_html=True)

            kpi2_text.write(
                f"<h3 style='text-align: center; color:red; '>{arahJalan}</h3>", unsafe_allow_html=True)

            kpi3_text.write(
                f"<h3 style='text-align: center; color:red; '>{speed}</h3>", unsafe_allow_html=True)

            kpi4_text.write(
                f"<h3 style='text-align: center; color:yellow; '>{width}</h3>", unsafe_allow_html=True)

            kpi5_text.write(
                f"<h3 style='text-align: center; color:yellow; '>{label}</h3>", unsafe_allow_html=True)

            y2_text.write(
                f"<h3 style='text-align: center; color:green; '>Receiving Data ..</h3>", unsafe_allow_html=True)

            x1_text.write(
                f"<h3 style='text-align: center; color:yellow; '>{Objheight}</h3>", unsafe_allow_html=True)

            y1_text.write(
                f"<h3 style='text-align: center; color:yellow; '>{Objwidth}</h3>", unsafe_allow_html=True)

            x2_text.write(
                f"<h3 style='text-align: center; color:yellow; '>{center_coor}</h3>", unsafe_allow_html=True)

            score_text.write(
                f"<h3 style='text-align: center; color:yellow; '>{pred_score}</h3>", unsafe_allow_html=True)

        stoplur = main_button.button(
            'STOP ENGINE', key='stopmang', on_click=store_stop)

        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            stframe.image(vis, channels='BGR', use_column_width=False)

            write_atributs()

            toArduino = ("<" + arahJalan + ", " + str(speed) + ">")

            arduino.write_data(toArduino)

            if stoplur:
                pass

        cam.release()


def main():
    st.set_page_config(
        page_title="Robot Web Monitoring",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    image = Image.open('..\..\photos_colab\logouny.png')
    columns = st.sidebar.columns((1, 1, 1))
    columns[1].image(image, width=130)

    columns2 = st.sidebar.columns((1, 1, 1))
    with columns2[1]:
        original_title = '<p style="text-align: center; font-size: 14px;"><strong>Neil Armstrong NIM.18502244004</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)

    st.title('Obstracle Avoidance Rover using Image Segmentation Mask RCNN')
    st.markdown('----')

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

    mode = ['Welcome!', 'Sistem Kendali']
    app_mode = st.sidebar.selectbox(
        '', mode)
    if app_mode == 'Welcome!':
        welcome()
    elif app_mode == 'Sistem Kendali':
        webcam()


if __name__ == "__main__":
    main()
