# Obstracle Avoidance Robot using Mask R-CNN Image Segmentation

<img src="https://raw.githubusercontent.com/ctensz65/InstanceSegmentation/main/photos_colab/IMG_3123.jpg">

The robot is designed in the form of a 4-wheel drive car and is equipped with a cutting-edge Mask R-CNN system for obstacle avoidance. The system is capable of recognizing different objects and responding accordingly with specific avoidance maneuvers. In total, there are three types of obstacles that have been programmed to elicit unique evasion strategies. To enhance the user experience, a web-based monitoring application has been developed, allowing real-time visualization of the robot's instance and detection data.

# Dependencies

- Microsoft Visual C++ Build Tools (For windows installation)
- PyTorch
- Detectron2
- Anaconda
- CUDA Toolkit
- cuDNN

# Installation

### 1. Create New Conda Environment

```
conda create -n detectron2 python=3.10
conda activate detectron2

# install pytorch
# check again your cuda version with pytorch compatibilty
conda install pytorch torchaudio cudatoolkit=11.3 -c
```

### 2. Install Requirements

```
git clone https://github.com/ctensz65/Obstracle-Avoidance-MaskRCNN.git
pip3 install -r requirements.txt
```

### 3. Install Juypter Notebook

```
conda activate detectron2
conda install juypter
```

### 4. Install Detectron2

```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

### 5. Install Streamlit

```
pip3 install streamlit
```

# Train

There are 3 objects that become obstacles in this project. I was using [LabelMe](https://github.com/wkentaro/labelme) on annotation image.

I've created Colab Notebook to help and simplify on training Mask R-CNN Model. With using colab, there is no need to prepare devices with better GPU.
[Colab Notebook](https://colab.research.google.com/drive/16vrmhOQQQ57TF1HTYZ7Hvblm2xW0IKE9?usp=share_link)

Otherwise, you still can train the object locally with the Notebook Main-Project-Notebook.ipynb

Also, I've attached the labeled images with two separate folders (train and val). Hope, it'll help.
[LabeledObject](https://drive.google.com/file/d/18CHMTPAPgW4E0bYBa1YE0qFrjik-afo4/view?usp=share_link)

# Demo Robot

Connect arduino to COM PORT
Run main app with streamlit

```
cd ObstacleAvoidance\src
streamlit run app.py
```

Access the localhost with given port of streamlit.

# References

1. https://github.com/facebookresearch/detectron2
2. https://github.com/ohyicong/Google-Image-Scraper
