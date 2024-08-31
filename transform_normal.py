import torch
import shutil
import random
import os
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt



destination_folder = 'train/Augmentations/CenterCropResize/'

image_rewrite_string = 'CenterCropResize'

transform = A.Compose(
    [
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180,p=1),
        A.CenterCrop(height=180, width=180,p=1),
        A.Resize(height=250, width=250, p=1)
    ]
)

pwd = 'train/normal'
img_list = os.listdir(pwd)


for file in img_list:
    img = cv2.imread(os.path.join(pwd,file))
    image = np.array(img)
    aug = transform(image=image)
    new_img = aug["image"]
    cv2.imwrite(destination_folder+image_rewrite_string+'_'+file,new_img) 
