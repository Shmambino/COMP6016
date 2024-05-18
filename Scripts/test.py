from classes import GoogLeNet_ft, GoogLeNet
from contextlib import redirect_stdout


import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision
from torchvision.transforms import v2  # Transformations we can perform on our dataset
from torchvision.io import read_image
import torchinfo
from torch.utils.data import Dataset

model = GoogLeNet_ft()
model2 = GoogLeNet()

counter = 0
total = 0

# print(len(model.layers.parameters()))
# checking that model intialised correctly
for param in model.layers.parameters():
    if param.requires_grad == True:
        counter += 1
        total += 1
    else:
        total += 1

percentage = counter / total
print(f"Total Params that are trainable: {counter}")


print(model.layers)
# print(model2.model)
