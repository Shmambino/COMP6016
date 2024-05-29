## Feature Extraction from CNN - script for Campbell Eaton [18824421] - COMP6016 - Final Project

from Scripts.classes_z import rp_dataset, GoogLeNet

import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
from torchvision.transforms import v2  # Transformations we can perform on our dataset
from torch.utils.data import DataLoader

import pandas as pd


def main():

    transform = v2.Compose(
        [
            v2.Resize(224),  # expects input size of 224x224 minimum
            v2.ToDtype(
                torch.float32, scale=True
            ),  # transform to float to be clipped between 0-1
            # v2.Normalize(
            #   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # ),  # normalisation values in googlenet paper
        ]
    )

    # healthy
    training_healthy_data_dir = "../RP/Train-Healthy"
    mapping = "../RP/Train-Healthy/mapping.csv"

    # dataset
    training_healthy = rp_dataset(
        mapping, training_healthy_data_dir, transform=transform
    )
    # dataloader
    training_healthy_dataloader = DataLoader(
        training_healthy, 256, shuffle=False, pin_memory=True
    )

    # unhealthy
    training_unhealthy_data_dir = "../RP/Train-Unhealthy"
    mapping = "../RP/Train-Unhealthy/mapping.csv"

    # dataset
    training_unhealthy = rp_dataset(
        mapping, training_unhealthy_data_dir, transform=transform
    )
    # dataloader
    training_unhealthy_dataloader = DataLoader(
        training_unhealthy, 256, shuffle=False, pin_memory=True
    )

    # intialise model
    model = GoogLeNet()

    # extract features from dataloaders
    healthy_features = model.feature_extraction(training_healthy_dataloader)
    unhealthy_features = model.feature_extraction(training_unhealthy_dataloader)

    # save to file
    pd.DataFrame.from_records(healthy_features).to_csv("../Data/healthy_googlenet.csv")
    pd.DataFrame.from_records(unhealthy_features).to_csv(
        "../Data/unhealthy_googlenet.csv"
    )


if __name__ == "__main__":
    main()
