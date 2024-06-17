import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from Unet import UNet
import torch
import torchvision
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

#-------------------------------------------------------------------
categories = {  0: [0, 1, 2, 3, 4, 5, 6],                   #void
                1: [7, 8, 9, 10],                           #flat
                2: [11, 12, 13, 14, 15, 16],                #construction
                3: [17, 18, 19, 20],                        #object
                4: [21, 22],                                #nature
                5: [23],                                    #sky
                6: [24, 25],                                #human
                7: [26, 27, 28, 29, 30, 31, 32, 33, -1]}    #vehicle

categories_colors = {   0: (147, 2, 134),
                        1: (50, 50, 50),
                        2: (158, 96, 0),
                        3: (83, 127, 21),
                        4: (140, 234, 5),
                        5: (3, 226, 241),
                        6: (255, 0, 0),
                        7: (243, 236, 8)}

#-------------------------------------------------------------------
class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model, mit_scale
        ):
        super(SegmentationModel, self).__init__()
        self.n_channels = 3
        self.n_classes = 8
        self.name = f"SegFormer_HuggingFace_mit{mit_scale}"
        self.model = model
        
        
    def forward(self, image):
        outputs = self.model(pixel_values=image)
        
        upsampled_logits = nn.functional.interpolate(
            outputs.logits,
            size=(512,512), 
            mode="bilinear",
            align_corners=False
        )
        
        return upsampled_logits

#-------------------------------------------------------------------
def convert_mask(mask, convert_categories = True) :
    new_mask = np.zeros((512, 512))
    if convert_categories:
        for k in range(0, 8):
            for i in range(-1, 34) :
                map_array = categories[k]
                if i in map_array :
                    for v in map_array: new_mask[mask==v] = k
        new_mask = new_mask.astype(int)
    else:
        new_mask = mask
    
    mask_colored = np.zeros((512, 512, 3))
    for k in range(0, 8):
        mask_colored[new_mask==k] = categories_colors[k]
    mask_colored = mask_colored.astype(int)
    return mask_colored

#-------------------------------------------------------------------
def load_distrib_df():
    distribution = pd.read_csv("./Models/distribution.csv")
    return distribution

#-------------------------------------------------------------------
def infer_to_mask(model, image_path) :
    image = torchvision.io.read_image(f'./Data/images/{image_path}')
    T = torchvision.transforms.Resize((512, 512), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    image = T(image)
    image = image.numpy() / 255.0
    image = np.squeeze(image)
    image = image[None, :, :, :]
    image = torch.as_tensor(image.copy()).float().contiguous()

    model.eval()
    with torch.no_grad():
        img = image.to(device=device, dtype=torch.float32)
        pred = model(img).cpu()
        mask = pred.argmax(dim=1)

    mask = mask.squeeze(0)
    mask = mask.numpy()

    mask_colored = convert_mask(mask, convert_categories=False)
    return mask_colored

#-------------------------------------------------------------------
#--------------------------- APP -----------------------------------
#-------------------------------------------------------------------
st.set_page_config(
    page_title="Image Segmentation",
    #layout="centered",
    layout="wide",
    initial_sidebar_state="expanded")

st.title("Segmentation d'image, Comparaison Unet et SegFormer")

st.header('Présentation du jeu de données')

st.write("Le jeu de donnée se compose d'images et de masks, contenant l'information de la classe à laquelle appartiennent les pixels :")

col = st.columns((4.5, 4.5, 2), gap='medium')

with col[0]:
    image = Image.open('./Data/images/cologne_000003_000019_leftImg8bit.png')
    image = image.resize((512,512))
    st.image(image)

with col[1]:
    mask = Image.open('./Data/masks/cologne_000003_000019_gtFine_labelIds.png').convert('L')
    mask = mask.resize((512,512), Image.Resampling.NEAREST)
    mask = np.array(mask.getdata()).reshape(mask.size[0], mask.size[1])
    mask_colored = convert_mask(mask)
    st.image(mask_colored)

st.header('Distribution des données:')
distribution = load_distrib_df()
fig1 = plt.figure(figsize=(4,4))
plt.pie(distribution['count'], labels=distribution['index'], autopct='%1.1f%%')
#plt.title("Total Categories distributions per pixels")
st.pyplot(fig1, use_container_width=False)

st.write("Nous sommes dans un cas où les données sont particulièrements désiquilibrés, nous utiliserons donc l'IoU (et la matrice de confusion) comme métrique principales pour comparer la qualité de la segmentation de nos 2 modèles.")

st.header("Historique d'entrainement:")

st.subheader('Unet')
image = Image.open('./Data/appImages/unet_trainHisto.png')
st.image(image)

st.subheader('SegFormer')
image = Image.open('./Data/appImages/b3_trainHisto.png')
st.image(image)

st.header('Résultats après entrainement:')
col = st.columns((4.5, 4.5, 2), gap='medium')

with col[0]:
    st.header('Unet')
    cm = Image.open('./Data/appImages/unet_cm.png')
    st.image(cm)

with col[1]:
    st.header('SegFormer b3')
    cm = Image.open('./Data/appImages/b3_cm.png')
    st.image(cm)

st.header('Test Inférence:')
file_list = os.listdir('./Data/images/')
option = st.selectbox('Selectionnez une image a inférer.', file_list)
device = "cpu"

# Load models
unet_model = UNet(n_channels=3, n_classes=8)
unet_model = unet_model.to(device)
checkpoint = torch.load(f"./Models/{unet_model.name}.pt", map_location=torch.device('cpu'))
unet_model.load_state_dict(checkpoint['model_state_dict'])

segFormer_huggingFace_b3 = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b3",
        ignore_mismatched_sizes=True, 
        num_labels=8, 
        reshape_last_stage=True
    )
model = SegmentationModel(segFormer_huggingFace_b3, "b3")
segformer = model.to(device)
checkpoint = torch.load(f"./Models/{segformer.name}.pt", map_location=torch.device('cpu'))
segformer.load_state_dict(checkpoint['model_state_dict'])

# inference
mask_unet = infer_to_mask(unet_model, option)
mask_seg = infer_to_mask(segformer, option)

col = st.columns((4.5, 4.5, 2), gap='medium')

with col[0]:
    st.header('Image')
    image = Image.open(f'./Data/images/{option}')
    image = image.resize((512,512))
    st.image(image)

with col[1]:
    st.header('Original Mask')
    mask_path = option.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
    mask = Image.open(f'./Data/masks/{mask_path}').convert('L')
    mask = mask.resize((512,512), Image.Resampling.NEAREST)
    mask = np.array(mask.getdata()).reshape(mask.size[0], mask.size[1])
    mask_colored = convert_mask(mask)
    st.image(mask_colored)

col = st.columns((4.5, 4.5, 2), gap='medium')

with col[0]:
    st.header('Unet')
    st.image(mask_unet)
    
with col[1]:
    st.header('SegFormer b3')
    st.image(mask_seg)