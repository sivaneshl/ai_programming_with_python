import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from collections import OrderedDict
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0,5))])

# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

image, label = next(iter(trainloader))
helper.imshow(image[0, :])

# Building the network
# Here you should define your network. As with MNIST, each image is 28x28 which is a total of 784 pixels, and there are
# 10 classes. You should include at least one hidden layer. We suggest you use ReLU activations for the layers and to
# return the logits from the forward pass. It's up to you how many layers you add and the size of those layers.

# Hyperparameters
input_size = 784
hidden_size = [256, 128, 64]
output_size = 10

# Build a feed forward network
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_size[0])),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(hidden_size[0], hidden_size[1])),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(hidden_size[1], hidden_size[2])),
    ('relu3', nn.ReLU()),
    ('logits', nn.Linear(hidden_size[2], output_size))
]))

# Train the network
# Now you should create your network and train it. First you'll want to define the criterion ( something like
# nn.CrossEntropyLoss) and the optimizer (typically optim.SGD or optim.Adam).
#
# Then write the training code. Remember the training pass is a fairly straightforward process:
#
# Make a forward pass through the network to get the logits
# Use the logits to calculate the loss
# Perform a backward pass through the network with loss.backward() to calculate the gradients
# Take a step with the optimizer to update the weights
# By adjusting the hyperparameters (hidden units, learning rate, etc), you should be able to get the training loss below 0.4.

# Create the network, define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 3
steps = 0
print_every = 40

# Train the network here
for e in range(epochs):
    running_loss = 0
    for images, labels in iter(trainloader):
        steps += 1
        # Flatten MNIST images into a 784 long vector
        images.resize_(32, 784)

        optimizer.zero_grad()

        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            print('Epoh {}/{}'.format(e+1, epochs),
                  'Loss: {:.4f}'.format(running_loss/print_every))
            running_loss = 0