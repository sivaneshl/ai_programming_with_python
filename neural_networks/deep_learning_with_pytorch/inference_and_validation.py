import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the testing data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

## Building the network
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feed forward network with arbitrary hidden layers
            Argments:
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output
            hidden_layers: list of integers, sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''

        super().__init__()

        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # Add the output layer -> last hidden layer to output
        self.output = nn.Linear(hidden_layers[-1], output_size)

        # Dropout probability
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        # The forward method is returning the log softmax for the output. The log-softmax is a log probability.
        # Using the log probability, computations are often faster and more accurate. To take the cross probabilities
        # later, we need to take the exponential (torch.exp) of the output.
        return F.log_softmax(x, dim=1)

## Train the network
# Create the network, define the criterion and optimizer
model = Network(784, 10, [512, 256, 128, 64], drop_p=0.5)
# Since the model returns log softmax, we need to use the negative log loss as criterion
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Implement a function for the validation pass
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        images.resize_(images.shape[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        probs = torch.exp(output)
        equality = (labels.data == probs.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

# Hyper parameters
epochs = 2
steps = 0
running_loss = 0
print_every = 40

# training
for e in range(epochs):
    model.train()   # training mode - turn ON dropout functionality
    for images, labels in trainloader:
        steps += 1

        # Flatten images into a 784 long vector
        images.resize_(images.shape[0], 784)

        optimizer.zero_grad()

        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, testloader, criterion)

            print('Epoch {}/{}'.format(e+1, epochs),
                  'Training Loss: {:.3f}'.format(running_loss/print_every),
                  'Test Loss: {:.3f}'.format(test_loss/len(testloader)),
                  'Accuracy: {:.3f}'.format(accuracy/len(testloader)))

            running_loss = 0
            model.train()   # training mode ON


## Testing the network
model.eval()

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.view(1, 784)

# Calculate the class probabilities (softmax) for img
with torch.no_grad():
    output = model.forward(img)

probs = torch.exp(output)

# Plot the image and probabilities
helper.view_clssify(img.view(1, 28, 28), probs, version='Fashion')
