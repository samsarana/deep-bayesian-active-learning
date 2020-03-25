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
        """
        Computes forward pass with input `x` and returns output

        Parameters
        ----------
        self
        x: torch.Tensor
            input tensor
            x.shape == [b, 1, 28, 28]
        
        Returns
        -------
        output: torch.Tensor
            output tensor
            output.shape == [b, 10]
        """
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

    def forward_stochastic(self, x, k=20):
        """
        Computes `k` stochastic forward passes with input `x` and returns tensor
        of all outputs
        NB Dropout must be on (model.train())

        Parameters
        ----------
        self
        x: torch.Tensor
            input tensor
            x.shape == [b, 1, 28, 28]
        k: int
            number of stochastic forward passes/dropout masks/samples from approx posterior
        
        Returns
        -------
        output: torch.Tensor
            output tensor
            output.shape == [b, 10, k]
        """
        self.train() # ensure dropout is on
        out = []
        for i in range(k):
            out.append(self.forward(x))
        return torch.stack(out, dim=-1)