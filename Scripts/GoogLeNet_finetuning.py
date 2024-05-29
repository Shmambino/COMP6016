from Scripts.classes_z import GoogLeNet_ft, rp_dataset
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


def main():

    transform = v2.Compose(
        [
            v2.Resize(224),  # expects input size of 224x224 minimum
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    data_dir = "../RP/Train-Healthy/distance_0.01/stride_100"
    mapping = "../RP/Train-Healthy/distance_0.01/stride_100/mapping.csv"

    # dataset
    full_dataset = rp_dataset(mapping, data_dir, transform=transform)

    # Training (80%)
    # 80% - sequential - class 0
    # 20% - shuffled - class 1
    #
    # Testing (20%)
    # 80% - sequential - class 0
    # 20% - shuffled - class 1
    #
    # Total
    # 100%

    # split dataset 80-20 (will be further split for sequential, and shuffled)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(
        full_dataset, range(train_size, train_size + test_size)
    )

    sequential_size = int(0.8 * len(train_dataset))
    shuffled_size = len(train_dataset) - sequential_size

    sequential_train_dataset = torch.utils.data.Subset(
        train_dataset, range(sequential_size)
    )
    shuffled_train_dataset = torch.utils.data.Subset(
        train_dataset, range(sequential_size, sequential_size + shuffled_size)
    )

    # force labels to be 0 for sequential data
    sequential_train_dataset.dataset.dataset.mapping.loc[0:sequential_size, 4] = 0

    print(sequential_train_dataset.dataset.dataset.mapping.loc[0:sequential_size, 4])
    print(train_dataset.dataset.mapping[4])

    """

    # force labels to be 1 for shuffled data (surrogate target)
    sequential_train_dataset.dataset.dataset.mapping[4][
        sequential_size : sequential_size + shuffled_size
    ] = 1

    # split test (20% of full) into 80 % sequential and 20% shuffled
    sequential_size = int(0.8 * len(test_dataset))
    shuffled_size = len(test_dataset) - sequential_size

    sequential_test_dataset = torch.utils.data.Subset(
        test_dataset, range(sequential_size)
    )
    shuffled_test_dataset = torch.utils.data.Subset(
        test_dataset, range(sequential_size, sequential_size + shuffled_size)
    )

    # force labels to be 0 for sequential data
    sequential_train_dataset.dataset.dataset.mapping[4][
        len(train_dataset) : len(train_dataset) + sequential_size
    ] = 0

    # force labels to be 1 for shuffled data (surrogate target)
    sequential_train_dataset.dataset.dataset.mapping[4][
        len(train_dataset)
        + sequential_size : len(train_dataset)
        + sequential_size
        + shuffled_size
    ] = 1

    # print debugging
    print("-----------------------")
    print(f"Total dataset size is {len(full_dataset)}")
    print(f"Total train size is {len(train_dataset)}")
    print(f"Total test size is {len(test_dataset)}")
    print("-----------------------")
    print(f"Total train_sequential size is {len(sequential_train_dataset)}")
    print(f"Total train_shuffled size is {len(shuffled_train_dataset)}")
    print(f"== {len(sequential_train_dataset) + len(shuffled_train_dataset)}")
    print("-----------------------")
    print(f"Total test_sequential size is {len(sequential_test_dataset)}")
    print(f"Total test_shuffled size is {len(shuffled_test_dataset)}")
    print(f"== {len(sequential_test_dataset) + len(shuffled_test_dataset)}")
    print("-----------------------")

    # train_set = ConcatDataset([sequential_train_dataset, shuffled_train_dataloader])

    # dataloaders
    sequential_train_dataloader = DataLoader(
        sequential_train_dataset,
        128,
        shuffle=False,
        pin_memory=True,
    )

    shuffled_train_dataloader = DataLoader(
        shuffled_train_dataset,
        128,
        shuffle=True,
        pin_memory=True,
    )

    sequential_test_dataloader = DataLoader(
        sequential_test_dataset,
        128,
        shuffle=False,
        pin_memory=True,
    )

    shuffled_test_dataloader = DataLoader(
        shuffled_test_dataset,
        128,
        shuffle=True,
        pin_memory=True,
    )

    # initialise model
    model = GoogLeNet_ft()

    # specify hyperparameters
    learning_rate = 2e-7
    batch_size = 128
    num_epochs = 100
    momentum = 0.1
    weight_decay = 1e-3

    # specify optimiser
    optimizer = torch.optim.Adam(
        model.model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # calculating weights for loss function to assist with class imbalance
    total = sequential_train_dataset.dataset.dataset.mapping[4].value_counts()

    pos_weight = total[0] / total[1]

    weights = [pos_weight]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_weight = torch.Tensor(weights).to(device)

    # specify loss functions
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for t in trange(num_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        print("\nTraining Sequential Data..")
        model.model_train(sequential_train_dataloader, optimizer, loss_fn, batch_size)
        print("\nTesting Sequential Data..")
        model.model_test(sequential_test_dataloader, loss_fn)
        print("\nTraining Shuffled Data..")
        model.model_train(shuffled_train_dataloader, optimizer, loss_fn, batch_size)
        print("\nTesting Shuffled Data..")
        model.model_test(shuffled_test_dataloader, loss_fn)
    """


if __name__ == "__main__":
    main()
