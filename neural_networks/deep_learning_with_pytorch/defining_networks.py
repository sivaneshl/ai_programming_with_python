import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
from . import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

### To create a model
## Method 1 - using class
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)

    #  Forward pass through the network, returns the output digits
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

model = Network()
print(model)

# Get weights and bias
print(model.fc1.weight)
print(model.fc1.bias)

# Set bias to all 0s
model.fc1.bias.data.fill_(0)
# Set weights to a normal distribution
model.fc1.weight.data.normal_(std=0.01)

# Forward pass
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels)
images.resize(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to not automatically get batch size

# Forward pass through the network
img_idx = 0
probs = model.forward(images[img_idx, :])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), probs)

## Using nn.Sequential
# Hyperparameters for our network
input_size = 784
hidden_size = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_size[0]),
                      nn.ReLU,
                      nn.Linear(hidden_size[0], hidden_size[1]),
                      nn.ReLU,
                      nn.Linear(hidden_size[1], output_size),
                      nn.Softmax(dim=1))

print(model)

# Access the weight and bias
print(model[2].weight)
print(model[2].bias)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize(64, 1, 784)
probs =  model.forward(images[0, :])
helper.view_classify(images[0].view(1, 28, 28), probs)


## Using OrderedDict
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_size[0])),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(hidden_size[0], hidden_size[1])),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(hidden_size[1], output_size)),
    ('softmax', nn.Softmax(dim=1))
]))

print(model)

# Access the weight and bias
print(model.fc2.bias)
print(model.fc2.weight)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize(64, 1, 784)
probs = model.forward(images[0, :])
helper.view_classify(images[0].view(1, 28, 28), probs)

# Exercise: Build a network to classify the MNIST images with three hidden layers. Use 400 units in the first hidden
# layer, 200 units in the second layer, and 100 units in the third layer. Each hidden layer should have a ReLU
# activation function, and use softmax on the output layer.
# hyper parameters
input_size = 784
hidden_sizes = [400, 200, 100]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize(64, 1, 784)
probs = model.forward(images[0, :])
helper.view_classify(images[0].view(1, 28, 28), probs)



