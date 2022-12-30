import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid

#neural net imports
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import math
from PIL import Image


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=2), # in_channels,out_channels, kernel_size, stride, padding
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3),
                        nn.Conv2d(32, 64, kernel_size=2), # in_channels,out_channels, kernel_size, stride, padding
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(64, 128, kernel_size=2), # don't have to set padding, PyTorch can handle it.
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((6, 6)))
        self.classifier = nn.Sequential(
                            nn.Linear(128*6*6, 1024),
                            nn.Dropout(p=0.2),
                            nn.ReLU(),
                            nn.Linear(1024, 256),
                            nn.Dropout(p=0.1),
                            nn.ReLU(),
                            nn.Linear(256, 10))
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
        
class number_detection():
    def __init__(self):
        self.cnn = CNN()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("cuda")
        else:
            self.device = torch.device("cpu")
            print("cpu")
        self.cnn.to(self.device)
        self.optimizer = optim.Adam(self.cnn.parameters())
        self.loss_fn = torch.nn.CrossEntropyLoss()

        train_df = pd.read_csv("mnist_train.csv")
        

        Y_data = np.array(train_df["label"])
        X_data = np.array(train_df.loc[:,"pixel0":]).reshape(-1,1,28,28)/255

        tensor_x = torch.from_numpy(X_data).float()
        tensor_y = torch.from_numpy(Y_data)

        my_dataset = TensorDataset(tensor_x,tensor_y)
        train_dataset, val_dataset = torch.utils.data.random_split(my_dataset, (int(len(X_data)*0.99),int(len(X_data)*0.01)))
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=64)
        
    def train(self):
        model=self.cnn
        optimizer=self.optimizer
        loss_fn=self.loss_fn 
        train_loader=self.train_loader 
        val_loader=self.val_loader 
        epochs=10 
        device=self.device

        for epoch in range(epochs):
            training_loss = 0.0
            valid_loss = 0.0
            model.train()   # training mode
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                training_loss += loss.data.item() * inputs.size(0)
            training_loss /= len(train_loader.dataset)
            
            model.eval()   # evaluation mode for test.
            num_correct = 0
            num_examples = 0
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                output = model(inputs)
                targets = targets.to(device)
                loss = loss_fn(output,targets)
                valid_loss += loss.data.item() * inputs.size(0)
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= len(val_loader.dataset)
            print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss, valid_loss, num_correct / num_examples))
    
    def test(self):
        test_df = pd.read_csv("mnist_test.csv")
        X_test = np.array(test_df).reshape(-1,1,28,28)/255
        tensor_test = torch.from_numpy(X_test).float()
        tensor_test = tensor_test.to(self.device)
        
        rows = 2
        cols = 5
        axes=[]
        fig=plt.figure(figsize=(15,8))

        for a in range(rows*cols):
            print(tensor_test[9+a].reshape(1,1,28,28).shape)
            axes.append( fig.add_subplot(rows, cols, a+1) )
            subplot_title=("Label:" + str(torch.argmax(self.cnn(tensor_test[9+a].reshape(1,1,28,28))).item()))
            axes[-1].set_title(subplot_title)  
            plt.imshow(np.array(tensor_test[9+a].cpu()).reshape(28,28))
        fig.tight_layout()    
        plt.show()

    def inference(self, img = None):
        self.ToTensor = transforms.Compose([
            transforms.ToTensor()
        ])

        tensor = Image.open("./xx.png").resize((28,28))
        tensor = self.ToTensor(tensor).unsqueeze(0)
        tensor = tensor[:, 0, :, :].unsqueeze(1)
        tensor = tensor.to(self.device)
        print(tensor.shape)
        print(str(torch.argmax(self.cnn(tensor)).item()))

nd = number_detection()
nd.train()
# nd.test()
nd.inference()






#================================================
# train_labels = train_df['label'].values
# train_images = (train_df.iloc[:,1:].values).astype('float32')
# test_images = (test_df.iloc[:,:].values).astype('float32')

# #Training and Validation Split
# train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels,
#                                                                      stratify=train_labels, random_state=123,
#                                                                      test_size=0.20)
# train_images = train_images.reshape(train_images.shape[0], 28, 28)
# val_images = val_images.reshape(val_images.shape[0], 28, 28)
# test_images = test_images.reshape(test_images.shape[0], 28, 28)

                                                                 
# #train
# train_images_tensor = torch.tensor(train_images)/255.0
# train_labels_tensor = torch.tensor(train_labels)
# train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)

# #val
# val_images_tensor = torch.tensor(val_images)/255.0
# val_labels_tensor = torch.tensor(val_labels)
# val_tensor = TensorDataset(val_images_tensor, val_labels_tensor)

# #test
# test_images_tensor = torch.tensor(test_images)/255.0

# train_loader = DataLoader(train_tensor, batch_size=16, num_workers=2, shuffle=True)
# val_loader = DataLoader(val_tensor, batch_size=16, num_workers=2, shuffle=True)
# test_loader = DataLoader(test_images_tensor, batch_size=16, num_workers=2, shuffle=False)
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
        
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2) 
#         )
        
#         self.linear_block = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(128*7*7, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(64, 10)
#         )
        
#     def forward(self, x):
#         x = self.conv_block(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_block(x)
        
#         return x




# conv_model = Net()
# optimizer = optim.Adam(params=conv_model.parameters(), lr=0.003)
# criterion = nn.CrossEntropyLoss()

# exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# if torch.cuda.is_available():
#     conv_model = conv_model.cuda()
#     criterion = criterion.cuda()

# def train_model(num_epoch):
#     conv_model.train()
#     exp_lr_scheduler.step()
    
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data = data.unsqueeze(1)
#         data, target = data, target
        
#         if torch.cuda.is_available():
#             data = data.cuda()
#             target = target.cuda()
            
#         optimizer.zero_grad()
#         output = conv_model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
        
#         # if (batch_idx + 1)% 100 == 0:
#         #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#         #         num_epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
#         #         100. * (batch_idx + 1) / len(train_loader), loss.data[0]))
            
# def evaluate(data_loader):
#     conv_model.eval()
#     loss = 0
#     correct = 0
    
#     for data, target in data_loader:
#         data = data.unsqueeze(1)
#         data, target = data, target
        
#         if torch.cuda.is_available():
#             data = data.cuda()
#             target = target.cuda()
        
#         output = conv_model(data)
        
#         loss += F.cross_entropy(output, target, size_average=False).data[0]

#         pred = output.data.max(1, keepdim=True)[1]
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
#     loss /= len(data_loader.dataset)
        
#     # print('\nAverage Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(
#     #     loss, correct, len(data_loader.dataset),
#     #     100. * correct / len(data_loader.dataset)))

# num_epochs = 25

# for n in range(num_epochs):
#     train_model(n)
#     evaluate(val_loader)



# def make_predictions(data_loader):
#     conv_model.eval()
#     test_preds = torch.LongTensor()
    
#     for i, data in enumerate(data_loader):
#         data = data.unsqueeze(1)
        
#         if torch.cuda.is_available():
#             data = data.cuda()
            
#         output = conv_model(data)
        
#         preds = output.cpu().data.max(1, keepdim=True)[1]
#         test_preds = torch.cat((test_preds, preds), dim=0)
        
#     return test_preds

# test_set_preds = make_predictions(test_loader)
# print(test_set_preds)