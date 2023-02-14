# Obstracle Avoidance Robot using Mask R-CNN Image Segmentation

<img src="https://raw.githubusercontent.com/ctensz65/InstanceSegmentation/main/photos_colab/IMG_3123.jpg">

The robot has a form like 4WD Car. Mask R-CNN system will guide the movement of the robot to evade obstracles. The robot have a particular response for each objects that detected by the system.

# Dependencies

- Microsoft Visual C++ (For windows installation)
- PyTorch
- Detectron2
- Anaconda

# Installation

### 1. Create new conda env

```
conda create -n detectron2 python=3.10
conda activate detectron2

# install pytorch
# check again your cuda version with pytorch compatibilty
conda install pytorch torchaudio cudatoolkit=11.3 -c
```

### 2. Install detectron2

Cloning detectron2 package

```
git clone https://github.com/facebookresearch/detectron2.git

python -m pip install -e detectron2
```

# Train

There are 3 objects that become obstacles in this project. I was using [LabelMe](https://github.com/wkentaro/labelme) on annotation image.

I've created Colab Notebook to help and simplify on training Mask R-CNN Model. With using colab, there is no need to prepare devices with better GPU.
| https://colab.research.google.com/drive/16vrmhOQQQ57TF1HTYZ7Hvblm2xW0IKE9?usp=share_link

Also, I've attached the labeled images with two separate folders (train and val). Hope, it'll help.
| https://drive.google.com/file/d/18CHMTPAPgW4E0bYBa1YE0qFrjik-afo4/view?usp=share_link

# Demo Robot

Connect arduino to COM PORT
Run main app with streamlit

```
streamlit run app.py
```

# References

1. https://github.com/facebookresearch/detectron2
2. https://github.com/ohyicong/Google-Image-Scraper
