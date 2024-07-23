import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
import h5py
import math
import os
import scipy.io as sio


class CNN3D(nn.Module):
    def __init__(self, is_training=True):
        super(CNN3D, self).__init__()
        self.is_training = is_training
        ########################## T #################################
        self.conv1_1 = nn.Conv3d(1, 32, (1, 3, 3), padding=(0,1,1), bias=True)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv3d(1, 32, (1, 5, 5), padding=(0,2,2), bias=True)
        self.relu1_2 = nn.ReLU()
        self.conv1_3 = nn.Conv3d(1, 32, (1, 7,7), padding=(0,3,3), bias=True)
        self.relu1_3 = nn.ReLU()
        self.conv1_4 = nn.Conv3d(32*3, 32, (1,1,1) , padding=0, bias=True)        
        self.relu1_4 = nn.ReLU()



        self.conv2_1 = nn.Conv3d(32, 64, (1, 3, 3), padding=(0,1,1), bias=True)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv3d(32, 64, (1, 5, 5), padding=(0,2,2), bias=True)
        self.relu2_2 = nn.ReLU()
        self.conv2_3 = nn.Conv3d(32, 64, (1, 7,7), padding=(0,3,3), bias=True)
        self.relu2_3 = nn.ReLU()
        self.conv2_4 = nn.Conv3d(64*3, 64, (1,1,1), padding=0, bias=True)        
        self.relu2_4 = nn.ReLU()




        self.conv3_1 = nn.Conv3d(64, 128, 3, padding=1, bias=False)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv3d(64, 128, 5, padding=2, bias=False)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv3d(64, 128, 7, padding=3, bias=False)
        self.relu3_3 = nn.ReLU()
        self.conv3_4 = nn.Conv3d(128*3, 128, 1, padding=0, bias=False)        
        self.relu3_4 = nn.ReLU()
        
        self.batchnorm1 = nn.BatchNorm3d(32)
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.batchnorm3 = nn.BatchNorm3d(64)

        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=1)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=1)

        self.fc1 = nn.Linear(492032, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)
        self.fc4 = nn.Linear(4,1)

        self.dropout = nn.Dropout(0.2)

        

    def forward(self, input):
        sides = []
        out1_1 = self.conv1_1(input)
        out1_1 = self.relu1_1(out1_1)
        out1_1 = self.batchnorm1(out1_1) 
        #print(out1_1.shape)
        out1_2 = self.conv1_2(input)
        out1_2 = self.relu1_2(out1_2)
        out1_2 = self.batchnorm1(out1_2)
        #print(out1_2.shape)
        out1_3 = self.conv1_3(input)
        out1_3 = self.relu1_3(out1_3)
        out1_3 = self.batchnorm1(out1_3)
        #print(out1_3.shape)
        out1_4 = torch.cat((out1_1, out1_2, out1_3), 1)
        #print(out1_4.shape)
        out1_4 = self.conv1_4(out1_4)
        out1_4 = self.relu1_4(out1_4)
        out1_4 = self.maxpool1(out1_4)
        out1_4 = self.batchnorm1(out1_4)
               

        
        out2_1 = self.conv2_1(out1_4)
        out2_1 = self.relu2_1(out2_1)
        out2_1 = self.batchnorm2(out2_1)
        out2_2 = self.conv2_2(out1_4)
        out2_2 = self.relu2_2(out2_2)
        out2_2 = self.batchnorm2(out2_2)
        out2_3 = self.conv2_3(out1_4)
        out2_3 = self.relu2_3(out2_3)
        out2_3 = self.batchnorm2(out2_3)
        out2_4 = torch.cat((out2_1, out2_2, out2_3), 1)
        out2_4 = self.conv2_4(out2_4)
        out2_4 = self.relu2_4(out2_4)  
        out2_4 = self.maxpool2(out2_4)       
        out2_4 = self.batchnorm2(out2_4) 
        
        # out3_1 = self.conv3_1(out2_4)
        # out3_1 = self.relu3_1(out3_1)
        # out3_2 = self.conv3_2(out2_4)
        # out3_2 = self.relu3_2(out3_2)
        # out3_3 = self.conv3_3(out2_4)
        # out3_3 = self.relu3_3(out3_3)
        # out3_4 = torch.cat((out3_1, out3_2, out3_3), 1)
        # out3_4 = self.conv3_4(out3_4)
        # out3_4 = self.relu3_4(out3_4)        

        out4_1 = torch.flatten(out2_4, 1)
        out4_2 = self.fc1(out4_1)
        if self.is_training:
            out4_2 = self.dropout(out4_2)
        out4_3 = self.fc2(out4_2)
        if self.is_training:
            out4_3 = self.dropout(out4_3)
        out4_4 = self.fc3(out4_3)
        if self.is_training:
            out4_4 = self.dropout(out4_4)
        out4_5 = self.fc4(out4_4)

        
        return out4_5