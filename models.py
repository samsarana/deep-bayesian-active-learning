import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianCNN(nn.Module):
    """
    From the paper: "acquisition functions are assessed with the same model structure:
    convolution-relu-convolution-relu-max pooling-dropout-dense-relu-dropout-dense-softmax,
    with 32 convolution kernels, 4x4 kernel size, 2x2 pooling, dense layer with
    128 units, and dropout probabilities 0.25 and 0.5 (following the example
    Keras MNIST CNN implementation (fchollet, 2015))."
    """
    def __init__(self): # TODO make nn.sequential, looks cleaner imo
        super(BayesianCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4) # params: (in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(7744, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # max pooling over 2x2 square window
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1) # don't flatten batch dimension
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output