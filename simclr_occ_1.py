import torch
import torch.nn as nn  
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from torchvision.datasets import ImageFolder
# from torchvision.transforms import (
#     RandomResizedCrop,
#     RandomHorizontalFlip,
#     ColorJitter,
#     RandomGrayscale,
#     RandomApply,
#     Compose,
#     GaussianBlur,
#     ToTensor,
#     Resize
# )
import torchvision.models as models

import os
import glob
import time
from skimage import io

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt

import cv2
from PIL import Image
import numpy as np


print(f'Torch-Version {torch.__version__}')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {DEVICE}')




class Config:
    root_path = 'train'
    number_augmentation = 1
    augmentation_types = ['CenterCropResize'] #'ColorJitter','ElasticTransform','RGBShift','CenterCropResize','GridDistortion']
    # augmentation_types = ['ElasticTransform'] #'ColorJitter','CenterCropResize','RGBShift','CenterCropResize','GridDistortion']  
    # augmentation_types = ['GridDistortion'] #'ColorJitter','CenterCropResize','RGBShift','CenterCropResize','GridDistortion']
    # augmentation_types = ['colorjitter'] #'ColorJitter','CenterCropResize','RGBShift','CenterCropResize','GridDistortion']
    # augmentation_types =['colorjitter','CenterCropResize']
    normal_path = os.path.join(root_path,'normal')
    aug_root = os.path.join(root_path,'Augmentations')
    #aug_path_dict = dict(augmentation_types,[os.path.join(aug_root ,x) for x in augmentation_types])
    
    
cfg = Config



def get_complete_transform(s=1.0):
    """
    The color distortion transform.

    Args:
        s: Strength parameter.

    Returns:
        A color distortion transform.
    """
    
    image_transform = A.Compose(
    [
        #A.VerticalFlip(p=0.5),
        #A.Rotate(limit=180,p=1),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1),
        #A.RandomSizedBBoxSafeCrop (150, 150, p=0.5),-
        A.RandomResizedCrop(224, 224, scale=(0.9, 1.0), ratio=(1, 1), interpolation=cv2.INTER_LANCZOS4, p=1.0), # Lanczos_is_the_best,writeinthepaper
        #A.Resize(height=250, width=250, p=1),
        ToTensorV2()
        
    ])
    
    return image_transform


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        views = [self.base_transform(x) for i in range(self.n_views)]
        return views


class CustomDataset(Dataset):

    def __init__(self,cfg, transform=None):
        """
        Args:
            list_images (list): List of all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cfg = cfg
        self.normal_list_image = os.listdir(cfg.normal_path)
        self.aug_list_dict = {}
        #normal_list_image
        #augmentation_list_image dictionary
        # for aug in cfg.augmentation_types:
        #     aug_list = os.listdir(os.path.join(cfg.aug_root,aug))
        #     self.aug_list_dict[aug] = aug_list
        self.transform = transform

    def __len__(self):
        return len(self.normal_list_image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_names = []
        images = []
        images_views = []
        
        normal_name = self.normal_list_image[idx]
        image_names.append(os.path.join(cfg.normal_path,normal_name))
        for aug in cfg.augmentation_types:
            # img_name = aug.lower() + '_'+normal_name  # remove lower in case of not colorjitter
            img_name = aug + '_'+normal_name  
            image_names.append(os.path.join(os.path.join(cfg.aug_root,aug),img_name))
        for img_path in image_names:
            #image = cv2.imread(img_path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.open(img_path)
            image = np.array(image)
            if self.transform:
                augmented = self.transform(image=image)
                image_first = augmented['image']
                second_augmented = self.transform(image=image)
                image_second = second_augmented['image']
            images.append(image_first)
            images_views.append(image_second)
            
        images = torch.stack(images)
        images_views = torch.stack(images_views)

        return images,images_views





train_set = CustomDataset(cfg,transform=get_complete_transform())
# train_set[0][0]





BATCH_SIZE = 64

def collate_fn(batch): 
    
    first_view = torch.stack([item[0] for item in batch])
    second_view = torch.stack([item[1] for item in batch])
    
    return [first_view,second_view]


train_dl = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
    collate_fn = collate_fn
)



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class SimCLR(nn.Module):
    def __init__(self, linear_eval=False):
        super().__init__()
        self.linear_eval = linear_eval
        resnet18 = models.resnet18(pretrained=False)
        resnet18.fc = Identity()
        self.encoder = resnet18
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
    def forward(self, x):
        if not self.linear_eval:
            x = torch.cat(x, dim=0)

        encoding = self.encoder(x)
        projection = self.projection(encoding)
        return projection



LABELS = torch.cat([torch.arange(len(cfg.augmentation_types)*BATCH_SIZE) for i in range(4)], dim=0)
LABELS = (LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float() # Creates a one-hot with broadcasting
LABELS = LABELS.to(DEVICE) #128,128

def cont_loss(features, temp):
    """
    The NTxent Loss.

    Args:
        z1: The projection of the first branch
        z2: The projeciton of the second branch

    Returns:
        the NTxent loss
    """
    similarity_matrix = torch.matmul(features, features.T) # 128, 128 # (B,B)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(LABELS.shape[0], dtype=torch.bool).to(DEVICE)
    # ~mask is the negative of the mask
    # the view is required to bring the matrix back to shape
    labels = LABELS[~mask].view(LABELS.shape[0], -1) # 1

    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # 128, 127

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) # 128, 1

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # 128, 126

    logits = torch.cat([positives, negatives], dim=1) # 128, 127
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE)

    logits = logits / temp
    return logits, labels
 
simclr_model = SimCLR().to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(simclr_model.parameters())



# Define the number of epochs
EPOCHS = 300

# Initialize an empty list to store losses
losses = []

# Define the path to save checkpoints
checkpoint_path = "./checkpoints_centercropresize_64"
os.makedirs(checkpoint_path, exist_ok=True)

# Initialize an unpretrained ResNet-18 model
simclr_model = models.resnet18(pretrained=False)
simclr_model.to(DEVICE)  # Assuming DEVICE is defined elsewhere in your code

# Define other necessary configurations like optimizer, loss function, etc.
# For example:
optimizer = optim.Adam(simclr_model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):  # Start from epoch 1
    t0 = time.time()
    running_loss = 0.0
    for i, views in enumerate(train_dl):
        bsz1, augsz, chn, w, h = views[0].shape

        views = [x.view(-1, chn, w, h) for x in views]
        views_concatenated = torch.cat([view.float().to(DEVICE) for view in views], dim=0)
        projections = simclr_model(views_concatenated)
        logits, labels = cont_loss(projections, temp=2)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss for the epoch
        running_loss += loss.item()

    # Store the average loss for the epoch
    epoch_loss = running_loss / len(train_dl)
    losses.append(epoch_loss)

    # Print loss and time taken for the epoch
    print(f'EPOCH: {epoch} LOSS: {epoch_loss:.16f}, Time: {(time.time() - t0) / 60:.4f} mins')

    # Save model checkpoints after every 50 epochs
    if epoch % 50 == 0:
        checkpoint_filename = f"checkpoint_epoch_{epoch}.pth"
        checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': simclr_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_filepath)
        print(f"Checkpoint saved at epoch {epoch} - {checkpoint_filepath}")

# Save the loss plot
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CenterCropResize - Batch Size 64 - Pretraining Loss Over Time')
plt.savefig('CenterCropResize_64_Pretraining_loss_plot.png')

# Save the final model
final_path = 'model_pretrained_centercropresize_batch64_final.pth'
torch.save({
    'epoch': EPOCHS,
    'model_state_dict': simclr_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, final_path)
print(f"Final model saved to {final_path}")



# checkpoint = torch.load(PATH)
# simclr_model.load_state_dict(checkpoint['model_state_dict'])


