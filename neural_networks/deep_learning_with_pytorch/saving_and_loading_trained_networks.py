from idlelib.browser import transform_children

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import helper
import fc_model

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the testing data
testset = datasets.FashionMNIST('F_MNIST_data/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

image, label = next(iter(trainloader))
helper.imshow(image[0, :])

## Train the network
# Create the network, define the criterion and optimizer
model = fc_model.Network(784, 10, [512, 256, 128])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Call the train method
fc_model.train(model, trainloader, testset, criterion, optimizer, epochs=2)

## Saving and Loading
print('Our Model: \n\n', model, '\n')
# The parameters for PyTorch networks are stored in a model's state_dict.
# We can see the state dict contains the weight and bias matrices for each of our layers.
print('The state dict keys: \n\n', model.state_dict().keys())

# The simplest thing to do is simply save the state dict with torch.save.
# For example, we can save it to a file 'checkpoint.pth'.
torch.save(model.state_dict(), 'checkpoint.pth')

# Then we can load the state dict with `torch.load`.
state_dict = torch.load('checkpoint.pth')
print(state_dict.keys())

# And to load the state dict in to the network, you do `model.load_state_dict(state_dict)`.
model.load_state_dict(state_dict)

# Information about the model architecture needs to be saved in the checkpoint, along with the state dict.
checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}
torch.save(checkpoint, 'checkpoint.pth')

# we can write a function to load checkpoints
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

model = load_checkpoint('checkpoint.pth')
print(model)