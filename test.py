import torch
from PIL import Image
import torchvision
from torchvision import transforms
import albumentations as a
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import numpy as np

A = a
# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "Data/Train_images/"
TRAIN_MASK_DIR = "Data/Train_masks/"
VAL_IMG_DIR = "Data/val_images/"
VAL_MASK_DIR = "Data/val_masks/"


val_transforms = transforms.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

def preprocess_image(image):
    image = np.array(image)[..., :3]
    image = Image.fromarray(image)
    # preprocess parameter
    CROP_SIZE=244
    MEAN=[0.0, 0.0, 0.0]
    STD=[1.0, 1.0, 1.0]

    # transform
    preprocess = transforms.Compose([
        transforms.Resize(size=(160, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    return preprocess(image).unsqueeze(dim=0)

def predict(img, model):
    img = preprocess_image(img)
    # #print(img.shape)
    # model.eval()
    # with torch.inference_mode():
    #     pred = model(img)
    #
    # return pred

    model.eval()

    with torch.no_grad():
        preds = torch.sigmoid(model(img))
        preds = (preds > 0.5).float()
    torchvision.utils.save_image(
        preds, "Predict/test.png"
    )
    #torchvision.utils.save_image(y.unsqueeze(1), "test.tif")

def main():
    model = UNET(in_channels=3, out_channels=1).to(torch.device('cpu'))

    checkpoint = torch.load("./my_checkpoint.pth.tar", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])

    xyz = Image.open("Data/val_images/TCGA_CS_6668_20011025_17.png")
    xyz = np.array(xyz)
    #xyz = preprocess_image(xyz)
    print(xyz.shape)

    predict(xyz, model)



if __name__ == "__main__":
    main()