# Imports
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import json
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image


def get_arguments():
    """ gets the command line arguments """

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='path to images folder')
    parser.add_argument('--save_dir', type=str, default='.', help='folder to save checkpoints')
    parser.add_argument('--arch', type=str, default='densenet121', help='model architecture - densenet121/vgg13')
    parser.add_argument('--hidden_units', type=int, default=512, help='# of hidden layer nodes')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--epochs', type=int, default=3, help='# of times to train')
    parser.add_argument('--gpu', type=bool, default=False, help='gpu or cpu')

    return parser.parse_args()


def load_datasets(data_dir='flowers'):
    """ loads the train, test and validation data sets """

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)

    return train_dataset, test_dataset, valid_dataset


def save_checkpoint(checkpoint, final=False):
    """ saves the checkpoint """

    save_dir = checkpoint['save_dir']
    file_name = (save_dir + '/checkpoint_final.pth' if final else save_dir + '/checkpoint.pth')
    print(file_name)
    torch.save(checkpoint, file_name)


def build_network(input_size, output_size, arch='densenet121', hidden_units=512, learning_rate=0.001, drop_p=0.5):
    """ builds the model and returns model, criterion and optimizer """

    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        raise TypeError('Not a supported model')

    # Freezing parameters so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Create a classifier that satisfies our purpose and replace the classifier in the model (pretrained densenet model)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(drop_p)),
        ('fc2', nn.Linear(hidden_units, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer


def eval_model(model, criterion, data_loader, device):
    """ evaluate the model """

    loss, accuracy = 0, 0
    model.eval()
    model.to(device)

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model.forward(images)
        loss += criterion(outputs, labels).item()

        probs = torch.exp(outputs).data
        equality = (labels.data == probs.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return loss, accuracy


def train_model(model, criterion, optimizer, train_loader, valid_loader, epochs, checkpoint, device):
    """ trains the model """

    print_every = 10
    running_loss = 0
    steps = 0

    for e in range(epochs):
        print('Training epoch ', e + 1)
        model.train()
        for images, labels in train_loader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    eval_loss, accuracy = eval_model(model, criterion, valid_loader, device)

                print('Epoch {}/{}'.format(e + 1, epochs),
                      'Training Loss: {:.3f}'.format(running_loss / print_every),
                      'Validation Loss: {:.3f}'.format(eval_loss / len(valid_loader)),
                      'Accuracy: {:.3f}'.format(accuracy / len(valid_loader)))

                running_loss = 0
                model.train()

        print('Saving checkpoint for epoch :', e + 1)
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        save_checkpoint(checkpoint)


# Main function
def main():
    # Get command line arguments
    cmd_args = get_arguments()
    data_dir = cmd_args.data_dir
    save_dir = cmd_args.save_dir
    arch = cmd_args.arch
    hidden_units = cmd_args.hidden_units
    learning_rate = cmd_args.learning_rate
    epochs = cmd_args.epochs
    gpu = cmd_args.gpu

    # Initialize device (cpu/gpu)
    device = (torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if gpu else 'cpu')

    # Load the datasets
    print('Loading Datasets...Start')
    train_dataset, test_dataset, valid_dataset = load_datasets(data_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
    print('Loading Datasets...Done')

    # Build the model
    print('Building Model...Start')
    drop_p = 0.5
    input_sizes = {"vgg13": 25088, "densenet121": 1024}
    input_size, output_size = input_sizes[arch], len(train_dataset.class_to_idx)
    model, criterion, optimizer = build_network(input_size, output_size,
                                                arch=arch,
                                                hidden_units=hidden_units,
                                                learning_rate=learning_rate)

    model.class_to_idx = train_dataset.class_to_idx
    model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    model.to(device)

    # Initializing check point
    checkpoint = {'arch': arch,
                  'input_size': input_size,
                  'hidden_units': hidden_units,
                  'output_size': output_size,
                  'dropout': drop_p,
                  'lr': learning_rate,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': train_dataset.class_to_idx,
                  'save_dir': save_dir}

    print('Building Model...Done')

    # Training
    print('Training...Start')
    #     train_model(model, criterion, optimizer, train_loader, valid_loader, epochs, checkpoint, device)
    print('Training...Done')

    # Save the checkpoint
    print('Saving final checkpoint...Start')
    checkpoint['state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    save_checkpoint(checkpoint, final=True)
    print('Saving final checkpoint...Done')

    # Testing
    print('Testing...Start')
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_loss, accuracy = eval_model(model, criterion, test_loader, device)
    print("Test Loss: {:.3f}..".format(test_loss / len(test_loader)),
          "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))
    print('Testing...Done')


# Main
if __name__ == "__main__":
    main()
