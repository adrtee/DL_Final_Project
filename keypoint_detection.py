import os
import json
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
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
            image = self.ToTensor(Image.open(self.imgs[idx]))
            return image, self.keypoints[idx]
        except:
            return None, None

class keypoint_detection():
    def __init__(self):
        self.img_dir = []
        self.keypoints = []
        self.dataset = None

    def import_data(self, parent):
        with open(parent+"train_GT_keypoints.json") as jsonfile:
            label_json = json.load(jsonfile)
        for i in label_json["annotations"]:
            self.keypoints.append(list(i.values())[-1])

        for i in label_json["images"]:
            self.img_dir.append(parent+"train_img/"+list(i.values())[0])

    def create_dataset(self):
        self.dataset = dataset_loader(self.img_dir, self.keypoints)
        
        tensor, kpt = self.dataset.__getitem__(0)
        print(tensor)
        print(kpt)


if __name__ == "__main__":
    keypoint_det = keypoint_detection()
    dataset_dir = "./crop_img/"
    keypoint_det.import_data(dataset_dir)
    keypoint_det.create_dataset()
    
