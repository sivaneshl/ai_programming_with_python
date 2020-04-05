# Imports
import argparse
import torch
import torchvision
import json
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image


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


def load_checkpoint(file_name='checkpoint_final.pth'):
    checkpoint = torch.load(file_name)

    model, criterion, optimizer = build_network(input_size=checkpoint['input_size'],
                                                output_size=checkpoint['output_size'],
                                                arch=checkpoint['arch'],
                                                hidden_units=checkpoint['hidden_units'],
                                                learning_rate=checkpoint['lr'],
                                                drop_p=checkpoint['dropout'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    return model, criterion, optimizer


def get_arguments():
    """ gets the command line arguments """

    parser = argparse.ArgumentParser()

    parser.add_argument('--image', type=str, default='flowers/test/10/image_07090.jpg',
                        help='path to the testing image')
    parser.add_argument('--topk', type=int, default=5, help='result top N classes')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth', help='checkpoint file')
    parser.add_argument('--category_labels', type=str, default='cat_to_name.json', help='category labels file')
    parser.add_argument('--gpu', type=bool, default=False, help='gpu or cpu')

    return parser.parse_args()


def process_image(image):
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return transform(Image.open(image))


def predict(image, model, device, topk=5):
    model.to(device)
    model.eval()

    img = process_image(image)
    img = img.unsqueeze_(0)
    img = img.float()

    with torch.no_grad():
        output = model.forward(img.to(device))

    prediction = torch.exp(output).data.topk(topk)
    probs = prediction[0].tolist()[0]
    classes = [model.idx_to_class[i] for i in prediction[1].tolist()[0]]

    return probs, classes


def main():
    # Get command line arguments
    cmd_args = get_arguments()
    test_image = cmd_args.image
    topk = cmd_args.topk
    checkpoint = cmd_args.checkpoint
    category_labels = cmd_args.category_labels
    gpu = cmd_args.gpu

    # Initialize device (cpu/gpu)
    device = (torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if gpu else 'cpu')

    # Load the model
    model, criterion, optimizer = load_checkpoint()

    # Load the categories json file
    with open(category_labels, 'r') as f:
        cat_to_name = json.load(f)

    # Process the image and predict
    probs, classes = predict(test_image, model, device)
    class_names = [cat_to_name[i] for i in classes]
    print(list(zip(classes, class_names, probs)))


if __name__ == '__main__':
    main()