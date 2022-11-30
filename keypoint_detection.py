import os
import json
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.utils import draw_keypoints
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2

class dataset_loader(Dataset):
    def __init__(self, imgs, keypoints):
        self.imgs = imgs
        self.keypoints = keypoints
        self.ToTensor = transforms.Compose([
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        try:
            image = read_image(self.imgs[idx])
            tensor = self.ToTensor(Image.open(self.imgs[idx]))
            return image, tensor, self.keypoints[idx]
        except:
            return None, None, None

class keypoint_detection():
    def __init__(self):
        self.img_dir = []
        self.keypoints = []
        self.dataset = None
        self.image = None
        self.tensor = None
        self.keypoint_label = None

    def import_data(self, parent):
        with open(parent+"train_GT_keypoints.json") as jsonfile:
            label_json = json.load(jsonfile)
        for i in label_json["annotations"]:
            self.keypoints.append(list(i.values())[-1])

        for i in label_json["images"]:
            self.img_dir.append(parent+"train_img/"+list(i.values())[0])

    def create_dataset(self):
        self.dataset = dataset_loader(self.img_dir, self.keypoints)

    def show_img_keypoint(self):
        self.image, self.tensor, self.keypoint_label = self.dataset.__getitem__(0)
        # 0,1: scale min
        # 2,3: scale max
        # 4,5: center
        # 6,7: pointer     
        res = draw_keypoints(self.image, torch.FloatTensor([[self.keypoint_label[0], self.keypoint_label[1]]]).unsqueeze(0), colors="blue", radius=5)
        res = draw_keypoints(res, torch.FloatTensor([[self.keypoint_label[2], self.keypoint_label[3]]]).unsqueeze(0), colors="blue", radius=5)
        res = draw_keypoints(res, torch.FloatTensor([[self.keypoint_label[4], self.keypoint_label[5]]]).unsqueeze(0), colors="red", radius=5)
        res = draw_keypoints(res, torch.FloatTensor([[self.keypoint_label[6], self.keypoint_label[7]]]).unsqueeze(0), colors="green", radius=5)
        transform = T.ToPILImage()
        img = transform(res)
        img.show()
if __name__ == "__main__":
    keypoint_det = keypoint_detection()
    dataset_dir = "./crop_img/"
    keypoint_det.import_data(dataset_dir)
    keypoint_det.create_dataset()
    keypoint_det.show_img_keypoint()
    
