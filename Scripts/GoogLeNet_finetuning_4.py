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

    transform = v2.Compose(
        [
            v2.Resize(224),  # expects input size of 224x224 minimum
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # sequential dataset
    data_dir = (
        "../RP/Train-Healthy/sliding_window_updated/distance_0.025/length_300/stride_30"
    )
    mapping = "../RP/Mappings/mapping_0.025d_300w_30s.csv"
    sequential_dataset = rp_dataset(mapping, data_dir, transform=transform)

    # shuffled dataset
    data_dir = "../RP/Train-Healthy/sliding_window_updated_shuffled/distance_0.025/length_300/stride_30"
    mapping = "../RP/Mappings/mapping_0.025d_300w_30s_shuffled.csv"
    shuffled_dataset = rp_dataset(mapping, data_dir, transform=transform)

    # Split both datasets into 80-20 Training and Testing
    # split dataset 80-20
    train_size = int(0.8 * len(sequential_dataset))
    test_size = len(sequential_dataset) - train_size

    train_dataset_sequential = torch.utils.data.Subset(
        sequential_dataset, range(train_size)
    )
    test_dataset_sequential = torch.utils.data.Subset(
        sequential_dataset, range(train_size, train_size + test_size)
    )

    train_dataset_shuffled = torch.utils.data.Subset(
        shuffled_dataset, range(train_size)
    )
    test_dataset_shuffled = torch.utils.data.Subset(
        shuffled_dataset, range(train_size, train_size + test_size)
    )

    # labelling
    # force labels to be 0 for sequential data
    sequential_dataset.mapping.loc[:, 4] = 0

    # force labels to be 1 for shuffled data
    shuffled_dataset.mapping.loc[:, 4] = 1

    print(len(train_dataset_sequential) == len(train_dataset_shuffled))
    print(len(test_dataset_sequential) == len(test_dataset_shuffled))

    # dataloaders
    sequential_train_dataloader = DataLoader(
        train_dataset_sequential,
        128,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    shuffled_train_dataloader = DataLoader(
        train_dataset_shuffled,
        128,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    sequential_test_dataloader = DataLoader(
        test_dataset_sequential,
        128,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    shuffled_test_dataloader = DataLoader(
        test_dataset_shuffled,
        128,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    # initialise model
    model = GoogLeNet_ft()

    # specify hyperparameters
    learning_rate = 1e-6
    batch_size = 128
    num_epochs = 100
    weight_decay = 1e-5

    # specify optimiser
    optimizer = torch.optim.Adam(
        model.model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # specify loss functions
    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_fn = nn.BCEWithLogitsLoss()

    best_loss = 0
    for t in trange(num_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        print("\nTraining..")
        model.model_train(
            sequential_train_dataloader,
            shuffled_train_dataloader,
            optimizer,
            loss_fn,
            batch_size,
        )
        print("\nTesting..")
        model.model_test(sequential_test_dataloader, shuffled_test_dataloader, loss_fn)

        if t == 1:
            model.test_loss[-1] = best_loss

        if t + 1 % 5 == 0:  # check every 5 epochs
            if model.test_loss[-1] > best_loss:
                print("Early Stopping!")
                break
            else:
                best_loss = min(model.test_loss[:])

    plt.plot(model.train_loss, label="Train")
    plt.plot(model.test_loss, label="Test")
    plt.legend()
    plt.savefig("train_test_loss_0.025d_300s_aux.png")
    torch.save(model, "../Models/best_model_0.025d_300s_aux")


if __name__ == "__main__":
    main()
