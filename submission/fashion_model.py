import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class Net(nn.Module):
    """
    Define your model here. Feel free to modify all code below, but do not change the class name. 
    This simple example is a feedforward neural network with one hidden layer.
    Please note that this example model does not achieve the required parameter count (101700). 
    """
    def __init__(self):
        super(Net, self).__init__()

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(48)

        # After convs + pool -> feature map is 48x12x12 = 6912
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(48 * 12 * 12, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
