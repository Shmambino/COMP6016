from classes import GoogLeNet_ft, rp_dataset
from contextlib import redirect_stdout


import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision
from torchvision.transforms import v2  # Transformations we can perform on our dataset
from torchvision.io import read_image
import torchinfo
from torch.utils.data import Dataset, DataLoader
from tqdm import trange


def main():

    transform = v2.Compose(
        [
            v2.Resize(224),  # expects input size of 224x224 minimum
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    training_healthy_data_dir = "../RP/Train-Healthy"
    mapping = "../RP/Train-Healthy/mapping.csv"

    # dataset
    training_healthy = rp_dataset(
        mapping, training_healthy_data_dir, transform=transform
    )
    # dataloader
    training_healthy_dataloader = DataLoader(
        training_healthy, 64, shuffle=False, pin_memory=True
    )

    # initialise model
    model = GoogLeNet_ft()

    # specify hyperparameters
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 1
    momentum = 0.1
    weight_decay = 1e-2

    # specify optimiser
    optimizer = torch.optim.Adam(
        model.layers.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # specify loss function
    loss_fn = nn.BCEWithLogitsLoss()

    for t in trange(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        model.model_train(training_healthy_dataloader, optimizer, loss_fn, batch_size)


if __name__ == "__main__":
    main()
