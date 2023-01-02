import os
from tqdm import tqdm
import json
import math
import torch
import operator
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.utils import draw_keypoints
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
from PIL import Image, ImageChops
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import cv2
from number_detection3 import number_detection
from statistics import mean

class dataset_loader(Dataset):
    def __init__(self, imgs, keypoints, input_size):
        self.imgs = imgs
        self.keypoints = keypoints
        self.input_size = input_size
        self.ToTensor = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        dir = self.imgs[idx].replace("./crop_img/train_img/", "")
        image = read_image(self.imgs[idx])
        tensor = Image.open(self.imgs[idx])
        # tensor = tensor.resize((self.input_size, self.input_size))
        tensor = self.ToTensor(tensor)

        # Normalise keypoints' coordinate
        kp = self.keypoints[idx]
        # kp = [element * self.input_size / image.shape[1] if index % 2 == 0 else\
        #       element * self.input_size / image.shape[2] for index, element in enumerate(kp)]

        # kp = [element if index % 2 == 0 else element * image.shape[1]/ image.shape[2] for index, element in enumerate(kp)]
        # kp = [element * self.input_size / image.shape[1] for index, element in enumerate(kp)]
        kp = torch.tensor(kp)
        
        return tensor, kp, dir

def get_model(num_kpts, train_kptHead=False, train_fpn=True):
    is_available = torch.cuda.is_available()
    device = torch.device('cuda:0' if is_available else 'cpu')
    dtype = torch.cuda.FloatTensor if is_available else torch.FloatTensor
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        pretrained=True, pretrained_backbone=True)

    for i, param in enumerate(model.parameters()):
        param.requires_grad = False

    if train_kptHead != False:
        for i, param in enumerate(model.roi_heads.keypoint_head.parameters()):
            if i/2 >= model.roi_heads.keypoint_head.__len__()/2-train_kptHead:
                param.requires_grad = True

    if train_fpn == True:
        for param in model.backbone.fpn.parameters():
            param.requires_grad = True

    out = nn.ConvTranspose2d(512, num_kpts, kernel_size=(
        4, 4), stride=(2, 2), padding=(1, 1))
    model.roi_heads.keypoint_predictor.kps_score_lowres = out
    print(model.roi_heads.keypoint_predictor)
    return model, device, dtype

class keypoint_detection():
    def __init__(self, img = None, kp = None):
        self.image = img
        self.keypoint = kp
        self.img_dir = []
        self.keypoints = []
        self.model, _, _ = get_model(4, train_kptHead=True, train_fpn=True)

        self.input_size = 256
        self.num_fold = 5
        self.epochs = 10
        self.batch_size = 1 # dont change
        self.optimizer = Adam(self.model.parameters(), lr=5e-5)
        self.mean_loss = []

        print("Pytorch version: " + torch.__version__)
        if torch.cuda.is_available():
            print("The model will be running on "+str(torch.cuda.get_device_name()))
        self.device = 'cuda:0'
        self.model.to(self.device) 
        
    def clear_output_folder(self):
        for file in os.listdir("./output_img"):
            file_path = os.path.join("./output_img", file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    def select_good_data(self):
        dir_dataset = "./crop_img/good_one"
        dir_list = next(os.walk(dir_dataset))
        return dir_list[2]
        # return ['scale_260_meas_0.png', 'scale_356_meas_0.png', 'scale_289_meas_0.png', 'scale_259_meas_0.png', 'scale_45_meas_0.png', 'scale_467_meas_0.png', 'scale_77_meas_0.png', 'scale_65_meas_0.png', 'scale_163_meas_0.png', 'scale_36_meas_0.png', 'scale_28_meas_0.png', 'scale_282_meas_0.png', 'scale_386_meas_0.png', 'scale_22_meas_0.png', 'scale_105_meas_0.png', 'scale_133_meas_0.png', 'scale_117_meas_0.png', 'scale_371_meas_0.png', 'scale_64_meas_0.png', 'scale_253_meas_0.png', 'scale_4_meas_0.png', 'scale_48_meas_0.png', 'scale_384_meas_0.png', 'scale_364_meas_0.png', 'scale_410_meas_0.png', 'scale_214_meas_0.png', 'scale_17_meas_0.png', 'scale_175_meas_0.png', 'scale_308_meas_0.png', 'scale_60_meas_0.png', 'scale_1_meas_0.png', 'scale_463_meas_0.png', 'scale_415_meas_0.png', 'scale_123_meas_0.png', 'scale_236_meas_0.png', 'scale_10_meas_0.png', 'scale_474_meas_0.png', 'scale_374_meas_0.png', 'scale_385_meas_0.png', 'scale_293_meas_0.png', 'scale_75_meas_0.png', 'scale_327_meas_0.png', 'scale_316_meas_0.png', 'scale_267_meas_0.png', 'scale_31_meas_0.png', 'scale_564_meas_0.png', 'scale_54_meas_0.png', 'scale_141_meas_0.png', 'scale_150_meas_0.png', 'scale_502_meas_0.png', 'scale_246_meas_0.png', 'scale_74_meas_0.png', 'scale_122_meas_0.png', 'scale_87_meas_0.png', 'scale_3_meas_0.png', 'scale_200_meas_0.png', 'scale_226_meas_0.png', 'scale_312_meas_0.png', 'scale_56_meas_0.png', 'scale_193_meas_0.png', 'scale_88_meas_0.png', 'scale_532_meas_0.png', 'scale_224_meas_0.png', 'scale_404_meas_0.png', 'scale_506_meas_0.png', 'scale_13_meas_0.png', 'scale_376_meas_0.png', 'scale_219_meas_0.png', 'scale_408_meas_0.png', 'scale_496_meas_0.png', 'scale_84_meas_0.png', 'scale_470_meas_0.png', 'scale_396_meas_0.png', 'scale_170_meas_0.png', 'scale_552_meas_0.png', 'scale_29_meas_0.png', 'scale_6_meas_0.png', 'scale_47_meas_0.png', 'scale_119_meas_0.png', 'scale_113_meas_0.png', 'scale_359_meas_0.png', 'scale_526_meas_0.png', 'scale_381_meas_0.png', 'scale_306_meas_0.png', 'scale_76_meas_0.png', 'scale_380_meas_0.png', 'scale_283_meas_0.png', 'scale_186_meas_0.png', 'scale_180_meas_0.png', 'scale_232_meas_0.png', 'scale_473_meas_0.png', 'scale_504_meas_0.png', 'scale_116_meas_0.png', 'scale_284_meas_0.png', 'scale_73_meas_0.png', 'scale_357_meas_0.png', 'scale_521_meas_0.png', 'scale_319_meas_0.png', 'scale_459_meas_0.png', 'scale_464_meas_0.png']
        
    def import_data(self, parent):
        with open(parent+"train_GT_keypoints.json") as jsonfile:
            label_json = json.load(jsonfile)

        for i in label_json["annotations"]:
            self.keypoints.append(list(i.values())[-1])

        for i in label_json["images"]:
            self.img_dir.append(parent+"train_img/"+list(i.values())[0])

        #sampling for faster training
        # self.keypoints = self.keypoints[:1]
        # self.img_dir = self.img_dir[:1]
        # self.keypoints = self.keypoints[5:6]
        # self.img_dir = self.img_dir[5:6]
  
    def import_data2(self, parent):
        good_img_dir = self.select_good_data()
        
        with open(parent+"train_GT_keypoints.json") as jsonfile:
            label_json = json.load(jsonfile)

        keypoints = []
        img_dir = []
        for i in label_json["annotations"]:
            keypoints.append(list(i.values())[-1])

        for i in label_json["images"]:
            img_dir.append(list(i.values())[0])
        
        for target in good_img_dir:
            index = img_dir.index(target)
            self.img_dir.append(parent+"train_img/"+target)
            self.keypoints.append(keypoints[index])

        # self.keypoints = self.keypoints[:10]
        # self.img_dir = self.img_dir[:10]
    
    def get_box(self, keypoints, size, x_shape, y_shape):
        # get radius
        center_x = [row[4] for row in keypoints]
        center_y = [row[5] for row in keypoints]
        point_tip_x = [row[6] for row in keypoints]
        point_tip_y = [row[7] for row in keypoints]
        
        radius_list = []
        top_left = []
        top_right = []
        bottom_left = []
        bottom_right = []
        for i in range(size):
            radius = math.sqrt((point_tip_x[i] - center_x[i]) ** 2 + (point_tip_y[i] - center_y[i]) ** 2)
            radius_list.append(radius)
        
            # top_left.append([center_x[i]-radius,center_y[i]-radius])
            top_right.append([center_x[i]+radius,center_y[i]-radius])
            bottom_left.append([center_x[i]-radius,center_y[i]+radius])
            # bottom_right.append([center_x[i]+radius,center_y[i]+radius])
            top_left.append(torch.zeros(2))
            bottom_right.append([torch.tensor(x_shape), torch.tensor(y_shape)])

        return top_left, top_right, bottom_left, bottom_right 

    def cross_val(self):
        kf = KFold(n_splits = self.num_fold)
        torch.cuda.empty_cache()
        for i, (train_index, val_index) in enumerate(kf.split(self.img_dir)): # np.array(self.training_set_raw['train_set_x'])
            print(f"Val set:  index = {val_index[0]} - {val_index[-1]}")
            
            x_train = [self.img_dir[i] for i in train_index] 
            x_val = [self.img_dir[i] for i in val_index]
            y_train = [self.keypoints[i] for i in train_index] 
            y_val = [self.keypoints[i] for i in val_index]
            
            self.mean_losses = []
            self.mean_val_losses = []
            self.train_acc = []
            self.val_acc = []

            training_set = dataset_loader(x_train, y_train, self.input_size)
            val_set = dataset_loader(x_val, y_val, self.input_size)

            self.train(training_set, i+1)
            self.validate(val_set)
        self.save_model()

    def train(self, training_set, fold):
        self.model.train()
        train_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=False)
        for epoch in range(1, self.epochs+1):
            print("-----")
            print("Current epoch: "+ str(epoch) + "; Current fold: "+ str(fold))
            epoch_loss = []
            index = 0
            for X, y, dir in tqdm(train_loader):
                x_shape = X.shape[2]
                y_shape = X.shape[3]
                top_left, top_right, bottom_left, bottom_right  = self.get_box(y, len(y), x_shape, y_shape)

                y = y.view(y.size(0),-1, 2)
                y_temp=torch.ones(y.shape[0],y.shape[1],1)
                y_temp=torch.cat([y,y_temp],2) #torch.Size([32, 4, 3])

                targets = []
                
                for i in range(len(y)):
                    box = [top_left[i][0].item(), top_left[i][1].item(), bottom_right[i][0].item(), bottom_right[i][1].item()]
                    box = torch.tensor(box).unsqueeze(0).to(self.device)
                    ones = torch.tensor([1]).to(self.device)
                    kp = y_temp[i].unsqueeze(0).to(self.device)
                    targets.append({'boxes': box,
                                    'labels': ones,
                                    'keypoints': kp })

                X=X.to(self.device) 

                output = self.model(X, targets=targets ) #{'loss_classifier': tensor(0.0027, device='cuda:0', grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0., device='cuda:0', grad_fn=<DivBackward0>), 'loss_keypoint': tensor(0., device='cuda:0', grad_fn=<MulBackward0>), 'loss_objectness': tensor(0.0283, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.0086, device='cuda:0', grad_fn=<DivBackward0>)}
                # print(output) 

                loss = output['loss_keypoint'] + output['loss_box_reg']
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()                
                epoch_loss.append(loss.item())

            self.mean_loss.append(mean(epoch_loss))
            print(f"Current Training Loss: {self.mean_loss[-1]}")
       
    def validate(self, val_set):
        self.model.eval()
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        for X, y, dir in tqdm(val_loader):
            X=X.to(self.device) 
            output = self.model(X)
            
            y = list(torch.unbind(y, dim=0))

            output_keypoints = []
            for i in range(len(output)):  # len shd be 1
                kp = output[i].get('keypoints')
                if output[i].get('keypoints_scores').numel() > 0: # outputs include a set of keypoints with confidence. Extract keypoints with highest confidence
                    _, max_scores_idx = torch.max(output[i].get('keypoints_scores'), dim=0)
                    kp0 = kp[max_scores_idx[0]][0][:-1]
                    kp1 = kp[max_scores_idx[1]][1][:-1]
                    kp2 = kp[max_scores_idx[2]][2][:-1]
                    kp3 = kp[max_scores_idx[3]][3][:-1]
                    kp = torch.cat((kp0, kp1, kp2, kp3), dim=0)
                else: 
                    kp = torch.tensor([0,0,0,0,0,0,0,0])
 
                output_keypoints.append(kp)
                self.show_img_keypoint(X[i], y[i], kp, dir[i])
                # print(kp)
                
    def show_img_keypoint(self, x, y, y_pred, dir):
        x = x*255
        x = x.byte()

        # 0,1: scale min
        # 2,3: scale max
        # 4,5: center
        # 6,7: pointer   
        # label y
        res = draw_keypoints(x,     torch.FloatTensor([[y[0], y[1]]]).unsqueeze(0), colors="blue", radius=5)
        res = draw_keypoints(res,   torch.FloatTensor([[y[2], y[3]]]).unsqueeze(0), colors="blue", radius=5)
        res = draw_keypoints(res,   torch.FloatTensor([[y[4], y[5]]]).unsqueeze(0), colors="blue", radius=5)
        res = draw_keypoints(res,   torch.FloatTensor([[y[6], y[7]]]).unsqueeze(0), colors="blue", radius=5)

        # prediction y_pred
        res = draw_keypoints(res, torch.FloatTensor([[y_pred[0], y_pred[1]]]).unsqueeze(0), colors="yellow", radius=5)
        res = draw_keypoints(res, torch.FloatTensor([[y_pred[2], y_pred[3]]]).unsqueeze(0), colors="yellow", radius=5)
        res = draw_keypoints(res, torch.FloatTensor([[y_pred[4], y_pred[5]]]).unsqueeze(0), colors="yellow", radius=5)
        res = draw_keypoints(res, torch.FloatTensor([[y_pred[6], y_pred[7]]]).unsqueeze(0), colors="yellow", radius=5)

        transform = T.ToPILImage()
        img = transform(res)
        # img.show()
        img.save('./output_img/'+dir)

    def plot_loss(self):
        """
        Plot graphs of losses
        """
        epochs = []
        for index, x in enumerate(self.mean_loss):
            epochs.append(index)

        plt.plot(epochs, self.mean_loss, label='Training')        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show() 

    def save_model(self):
        """
        save model parameters into pth
        """
        try: 
            PATH = "./model.pth"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
                }, PATH)
            print("Successfully saved model to "+PATH)
        except:
            print("Something went wrong! Saving failed.")
    
    def load_model(self):
        """
        load model parameters into model
        """
        PATH = "./model.pth"
        checkpoint = torch.load(PATH)
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Successfully loaded model checkpoint from "+str(PATH))
        except:
            import sys
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno
            print("Exception type: ", exception_type)
            print("File name: ", filename)
            print("Line number: ", line_number)
            print("Warning: pretrained model could not be loaded.")

    def inference(self):
        self.model.eval()
        test_set = dataset_loader(self.img_dir, self.keypoints, self.input_size)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        for X, y, dir in tqdm(test_loader):     
            X=X.to(self.device) 
            output = self.model(X)

            y = list(torch.unbind(y, dim=0))
            output_keypoints = []
            for i in range(len(output)): # len shd be 1
                kp = output[i].get('keypoints')
                if output[i].get('keypoints_scores').numel() > 0:
                    _, max_scores_idx = torch.max(output[i].get('keypoints_scores'), dim=0)
                    kp0 = kp[max_scores_idx[0]][0][:-1]
                    kp1 = kp[max_scores_idx[1]][1][:-1]
                    kp2 = kp[max_scores_idx[2]][2][:-1]
                    kp3 = kp[max_scores_idx[3]][3][:-1]
                    kp = torch.cat((kp0, kp1, kp2, kp3), dim=0)
                else: 
                    kp = torch.tensor([0,0,0,0,0,0,0,0])
 
                output_keypoints.append(kp)

                self.show_img_keypoint(X[i], y[i], kp, dir[i])
                self.get_numbers(X[i], kp)
                break
            # self.get_numbers(X[0].cpu().detach().numpy(), kp.cpu().detach().numpy())
            break

    def get_numbers(self, image, y_pred = None):
        transform = T.ToPILImage()
        image = np.array(image.cpu())*255
        image = torch.from_numpy(image)
        y_pred = np.array(y_pred.detach().cpu())

        for i in range(1):
            y = self.keypoints[i]
            img = transform(image)
            img = ImageChops.invert(img)
            img.show()

            min_box, max_box = self.get_small_box(img, y) # y_pred
            min_box.show()
            max_box.show()

            nd = number_detection(np.array(min_box))
            nd.preprocessing()
            max_num, min_num  = nd.get_text()
            print(min_num)
            print("---------------")
            nd = number_detection(np.array(max_box))
            nd.preprocessing()
            max_num, min_num  = nd.get_text()
            print(max_num)
            break


    def get_small_box(self, image, keypoints):
        min_x = keypoints[0] 
        min_y = keypoints[1] 
        max_x = keypoints[2] 
        max_y = keypoints[3] 
        # center_x = [row[4] for row in keypoints]
        # center_y = [row[5] for row in keypoints]
        # point_tip_x = [row[6] for row in keypoints]
        # point_tip_y = [row[7] for row in keypoints]

        size = 50

        min_box = image.crop((min_x-size, min_y-size, min_x+size, min_y+size))
        max_box = image.crop((max_x-size, max_y-size, max_x+size, max_y+size))
        return min_box, max_box

if __name__ == "__main__":
    dataset_dir = "./crop_img/"
    keypoint_det = keypoint_detection()       
    keypoint_det.import_data2(dataset_dir)
    
    # # train - val
    keypoint_det.clear_output_folder() 
    # keypoint_det.cross_val()
    # keypoint_det.plot_loss()

    # # test
    keypoint_det.load_model()
    keypoint_det.inference()
    # keypoint_det.get_numbers()


    #TODO:
    # use test set on inference
    # expand dataset
    # get number prediction done
    # calculate final value
