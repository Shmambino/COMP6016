from classes import GoogLeNet_ft, rp_dataset
from contextlib import redirect_stdout


import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision
from torchvision.transforms import v2  # Transformations we can perform on our dataset
from torchvision.io import read_image
import torchinfo
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
from tqdm import trange
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():

    model = torch.load("../Models/best_model.pth")

    plt.plot(model.train_loss[1::2], label="Train")
    plt.plot(model.test_loss[1::2], label="Test")
    plt.legend()
    plt.savefig("train_test_loss2.png")


if __name__ == "__main__":
    main()
