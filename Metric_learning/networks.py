import torch
import torch.nn as nn
import torch.nn.functional as F
import losses

############ BUILD THE NEURAL NETWORK ARCHITECTURES ###################


class LinearNetwork(nn.Module):
    def __init__(self, emb_dim = 2):
        super(LinearNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1024),  # the class nn.Linear directly initializes the parameters
            # with XAVIER INITIALIZATION
            nn.ReLU(), # rectifier activation function ReLu(x) = max(0,x)
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        F = self.linear_relu_stack(x)
        return F

# convolutional network of the paper Dimensionality Reduction by Learning an Invariant Mapping
# taking into consideration that the input image dim in the paper is 32x32 while in our dataset is 28x28

class CNN(nn.Module):
    def __init__(self, emb_dim = 2):
        super(CNN, self).__init__()
        self.conv = nn.Sequential( # stride = 1 and padding= 0 default values
            nn.Conv2d(1, 15, kernel_size=5, stride=1, padding=0),  # (in_channels, out_channels, kernel_size)
            # resulting shape of the layer: 15x24x24
            nn.ReLU(), # activation function rectifier
            nn.MaxPool2d(3), # average pooling with 3x3 kernel
            nn.Conv2d(15, 30, kernel_size=8, stride=1, padding=0),
            # resulting shape of the layer: 30x1x1
            nn.ReLU()
        )

        self.fc = nn.Sequential(  # fully connected layers
            nn.Linear(30, emb_dim),
            nn.Sigmoid()
        )
    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28) # reshape correctly the input batch of the CNN layer
        xb = self.conv(xb)
        xb = xb.view(-1, 30)
        xb = self.fc(xb)
        return xb

######## CNN MADE FRO SKIN CANCER DATA: INPUT SIZE 3x224x224

class CNN_SKIN(nn.Module):
    def __init__(self, emb_dim = 2):
        super(CNN_SKIN, self).__init__()
        self.conv = nn.Sequential( # stride = 1 and padding= 0 default values
            nn.Conv2d(3, 25, kernel_size=5, stride=1, padding=0),  # (in_channels, out_channels, kernel_size)
            # resulting shape of the layer: 15x24x24
            nn.ReLU(), # activation function rectifier
            nn.MaxPool2d(2), # average pooling
            nn.Conv2d(25, 50, kernel_size=5, stride=1, padding=0),  # (in_channels, out_channels, kernel_size)
            # resulting shape of the layer:
            nn.ReLU(),  # activation function rectifier
            nn.MaxPool2d(2),
            nn.BatchNorm2d(50),
            nn.Conv2d(50, 70, kernel_size=5, stride=1, padding=0),  # (in_channels, out_channels, kernel_size)
            # resulting shape of the layer: 70x24x24
            nn.ReLU(),  # activation function rectifier
            nn.MaxPool2d(2),
            nn.BatchNorm2d(70),
        )

        self.fc = nn.Sequential(  # fully connected layers
            nn.Linear(40320, emb_dim),
            nn.Sigmoid()
        )

    def forward(self, xb):
        xb = xb.view(-1, 3, 224, 224) # reshape correctly the input batch of the CNN layer
        xb = self.conv(xb)
        xb = xb.view(-1, 40320)
        xb = self.fc(xb)
        return xb