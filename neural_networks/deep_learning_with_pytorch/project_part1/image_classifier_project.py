import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image

# LOAD THE DATA SETS
data_dir = 'flowers'
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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# CREATE THE NETWORK
def build_network(hidden_size, input_size=1024, output_size=102, drop_p=0.5, learn_rate=0.001):
    model = models.densenet121(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Create a classifier that satisfies our purpose and replace the classifier in the model (pretrained densenet model)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_size)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(drop_p)),
        ('fc2', nn.Linear(hidden_size, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    return model, criterion, optimizer

model, criterion, optimizer = build_network(hidden_size=512)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Hyper parameters
epochs = 3
print_every = 10
steps = 0
running_training_loss = 0

# VALIDATION FUCNTION
def validation(model, data_loader, criterion, device):
    test_loss = 0
    accuracy = 0
    model.to(device)

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        probs = torch.exp(output).data
        equality = (labels.data == probs.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


# TRAINING
for e in range(epochs):
    model.train()
    for images, labels in train_loader:
        steps += 1
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_training_loss += loss.item()

        if steps % print_every == 0:
            model.eval()

            with torch.no_grad():
                test_loss, accuracy = validation(model, valid_loader, criterion, device)

            print('Epoch {}/{}'.format(e + 1, epochs),
                  'Training Loss: {:.3f}'.format(running_training_loss / print_every),
                  'Validation Loss: {:.3f}'.format(test_loss / len(valid_loader)),
                  'Accuracy: {:.3f}'.format(accuracy / len(valid_loader)))

            running_training_loss = 0
            model.train()

    print('Saving epoch :', e + 1)
    checkpoint = {'input_size': 1024,
                  'hidden_size': 512,
                  'output_size': 102,
                  'dropout': 0.5,
                  'lr': 0.001,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': train_dataset.class_to_idx}
    torch.save(checkpoint, 'checkpoint.pth')

print('Done!!')

# TESTING
model.to(device)
model.eval()

with torch.no_grad():
    test_loss, accuracy = validation(model, test_loader, criterion, device)

print("Test Loss: {:.3f}..".format(test_loss / len(test_loader)),
      "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))

# SAVE CHECKPOINT
model.class_to_idx = train_dataset.class_to_idx
model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}
checkpoint = {'input_size': 1024,
              'hidden_size': 512,
              'output_size': 102,
              'dropout': 0.5,
              'lr': 0.001,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}
torch.save(checkpoint, 'checkpoint_1.pth')

# LOAD CHECKPOINT
def load_checkpoint():
    checkpoint = torch.load('checkpoint_1.pth')
    model, criterion, optimizer = build_network(input_size=checkpoint['input_size'],
                                                output_size=checkpoint['output_size'],
                                                hidden_size=checkpoint['hidden_size'],
                                                drop_p=checkpoint['dropout'],
                                                learn_rate=checkpoint['lr'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    return model, criterion, optimizer


model, criterion, optimizer = load_checkpoint()

# PROCESS IMAGE
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return transform(Image.open(image))


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

test_image = process_image('flowers/test/10/image_07090.jpg')
imshow(test_image)

# CLASS PREDICTION
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # DONE: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()

    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()

    with torch.no_grad():
        output = model.forward(img.cuda())

    prediction = torch.exp(output).data.topk(topk)
    print(prediction)
    probs = prediction[0].tolist()[0]
    classes = [model.idx_to_class[i] for i in prediction[1].tolist()[0]]

    return probs, classes

predict('flowers/test/2/image_05100.jpg', model)

# SANITY CHECK
def sanity_check(image_path):
    fig, ax = plt.subplots()
    img = process_image(image_path)

    probs, classes = predict(image_path, model)
    class_names = [cat_to_name[i] for i in classes]
    print(probs, classes, class_names)
    ax.barh(class_names, probs)
    imshow(img)
    plt.show()

sanity_check('flowers/test/2/image_05100.jpg')