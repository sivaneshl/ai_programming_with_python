import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import helper

data_dir = 'Cat_Dog_data'

train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(100),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(244),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5],
                                                          [0.5, 0.5, 0.5])])

train_dataset = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
test_dataset = datasets.ImageFolder(data_dir + '/test', transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

data_iter = iter(test_loader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10, 4), ncols=4)
for p in range(4):
    ax=axes[p]
    helper.imshow(images[p], ax=ax)

plt.show()