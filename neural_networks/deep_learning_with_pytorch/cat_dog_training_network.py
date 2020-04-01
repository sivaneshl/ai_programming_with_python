import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import helper

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Download the images data from https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip
data_dir = 'Cat_Dog_data'

# Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_dataset = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_dataset = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# We can load in a model such as DenseNet.
model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad=False

# Create a classifier that satisfies our purpose and replace the classifier in the model (pretrained densenet model)
classifier = nn.Sequential(OrderedDict([
    # Classifier layer of densenet has 1024 input, that is got as input for our classifier ans is passed to 500 hidden
    # layers and then 2 output classes - cats and dogs
    ('fc1', nn.Linear(1024, 500)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(500, 2)),
    ('output', nn.LogSoftmax(dim=1))
]))
model.classifier = classifier

## Train the network
# Create the network, define the criterion and optimizer
# Since the model returns log softmax, we need to use the negative log loss as criterion
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001) # here we are only going to train the classifier part

# Hyper parameters
epochs = 1
print_every = 40
steps = 0
running_trainig_loss = 0

# Convert the model to cpu or gpu as per device
model.to(device)

# Implement a function for the validation pass
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        probs = torch.exp(output)
        equality = (labels.data == probs.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


# Training
model.train()
for e in range(epochs):
    model.train()
    for images, labels in train_loader:
        steps += 1
        # Move input and label tensors to the GPU/CPU
        images, labels = images.to(device), labels.to(device)

        # Flatten images into a 784 long vector
        # images.resize_(images.shape[0], (224*224))

        model.zero_grad()

        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_trainig_loss += loss.item()

        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, test_loader, criterion)

            print('Epoch {}/{}'.format(e + 1, epochs),
                  'Training Loss: {:.3f}'.format(running_trainig_loss / print_every),
                  'Test Loss: {:.3f}'.format(test_loss / len(test_loader)),
                  'Accuracy: {:.3f}'.format(accuracy / len(test_loader)))

            running_trainig_loss = 0
            model.train()

# Testing
model.eval()
test_loss = 0
accuracy = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        probs = torch.exp(output)
        equality = (labels.data == probs.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

print('Accuracy: {:.4f}'.format(accuracy))
print('Test loss: {:.4f}'.format(test_loss))
