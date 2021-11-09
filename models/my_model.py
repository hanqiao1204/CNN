

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv_layer = nn.Sequential(

            # Convolution stack 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolution stack 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout2d(p=0.05),

            # Convolution stack 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=7, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)

           

        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        N = x.shape[0]
        layer1 = self.conv_layer(x)
        outs = self.fc_layer(layer1.view(N, -1))
       
        
        return outs
