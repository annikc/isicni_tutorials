import numpy as np
from math import floor
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self,obs_shape, action_space):
        super(Net, self).__init__()
        self.input_channels = obs_shape[0] # (3x10x10)

        self.conv1 = nn.Conv2d(self.input_channels, 256, (3,3)) # output = 256x8x8
        output_shape = self.conv_output_shape(obs_shape[1:],kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(2) # output = 256x7x7
        output_shape = self.conv_output_shape(output_shape,kernel_size=2,stride=2)
        self.drop1 = nn.Dropout2d(p=0.2)

        self.conv2 = nn.Conv2d(256, 256, (3,3)) # output = 256x5x5
        output_shape = self.conv_output_shape(output_shape,kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(2) # output = 256x4x4
        output_shape = self.conv_output_shape(output_shape,kernel_size=2,stride=2)
        self.drop2 = nn.Dropout2d(p=0.2)

        print(f'output after conv2 is {output_shape}')

        self.fc1 = nn.Linear(output_shape[0] * output_shape[1] * 256, 64)  # why 3*3????
        self.fc2 = nn.Linear(64, action_space)

        self.loss_fxn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.drop2(x)

        # flatten inputs
        x = x.view(-1, self.num_flat_features(x))

        # pass through fully connected layers
        x = self.fc1(x)
        output = self.fc2(x)

        return output

    def conv_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
        w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
        return h, w


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class DeepQNetwork(nn.Module):
    def __init__(self,ALPHA): #ALPHA is the learning rate
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1,32,8, stride=4, padding=1)  #reduce computational requirements by using grayscale so one input channel needed
        self.conv2 = nn.Conv2d(32,64,4, stride=2)
        self.conv3 = nn.Conv2d(64,128,3)

        self.fc1 = nn.Linear(128*19*8, 512) ## dims just are that way for the atari inputs
        self.fc2 = nn.Linear(512, 6)

        self.optimizer = optim.RMSprop(self.parameters(),lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,observation):
        observation = T.Tensor(observation).to(self.device)
        # reshape to put num channels in first axis
        x = observation.view(-1, 1, 185, 95) #batch size, num channels, H, W
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128*19*8) # dims dependent on convolutions

        x = F.relu(self.fc1(x))
        actions = self.fc2(x)

        return actions
    



################3
input_shape = (3,10,10)
num_actions = 4
net_basic = Net(input_shape,num_actions)


input = torch.rand((1,)+input_shape)
output = net_basic(input)
print(output)