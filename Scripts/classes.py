## Class declarations for Campbell Eaton [18824421] - COMP6016 - Final Project

from functions import (
    check_conversion_pl,
    pivot_pl,
    drop_pl,
    resample_pl,
    reshape_input,
    scale_input,
    plot_reccurence,
)

import polars as pl
import polars.selectors as cs
import pandas as pd
import tqdm
from tqdm import trange

import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision
from torchvision.transforms import v2  # Transformations we can perform on our dataset
from torchvision.io import read_image
import torchinfo
import os
from torch.utils.data import Dataset
import numpy as np


class dataset:
    """
    This class provides functionality to create datasets with parameters provided to __init__
    """

    def __init__(
        self,
        raw_data: pl.DataFrame,
        n_rows: int,
        stride: int,
        iterations: int,
        target: str,
    ):
        self.raw_data = raw_data
        self.data = None
        self.n_rows = n_rows
        self.stride = stride
        self.iterations = iterations
        self.start = 0
        self.data_idx = 0
        self.final_df = None
        self.first = True
        self.dates = None
        self.array = None
        self.intermediate_feature_df = None
        self.final_feature_df = None
        self.curr_iteration = 0
        self.target = target

    def transform(self):
        """
        Performs data preprocessing functions for number of iterations specified in __init__
        """
        for i in trange(self.iterations):
            self.curr_iteration = i
            self.data = self.raw_data.slice(offset=self.start, length=self.n_rows)
            self.start = self.n_rows

            try:
                self._check_conversion_pl()
            except:
                raise ValueError("Dataframe contains null values")

            self.data = self._pivot_pl()
            self.data = self._resample_pl()
            self.data = self._drop_pl()
            self.dates = self.data.select(cs.datetime())
            self.array = self._reshape_input()
            self.array = self._scale_input()

            if self.first == True:
                self.final_feature_df, self.data_idx = self._plot_recurrence()
                self.first = False
            else:
                self.intermediate_feature_df, self.data_idx = self._plot_recurrence()
                self.final_feature_df = pl.concat(
                    [self.final_feature_df, self.intermediate_feature_df]
                ).lazy()

    def _check_conversion_pl(self):
        return check_conversion_pl(self.data)

    def _pivot_pl(self):
        return pivot_pl(self.data)

    def _resample_pl(self):
        return resample_pl(self.data)

    def _drop_pl(self):
        return drop_pl(self.data)

    def _reshape_input(self):
        return reshape_input(self.data.select(~cs.datetime()))

    def _scale_input(self):
        return scale_input(self.array)

    def _plot_recurrence(self):
        return plot_reccurence(
            self.array,
            self.stride,
            self.curr_iteration,
            self.dates,
            self.data_idx,
            target=self.target,
        )

    def __repr__(self) -> str:
        return print(self.raw_data.collect())

    def out_array(self):
        return self.array

    def out_features(self):
        return self.final_feature_df

    def out_raw(self):
        return self.raw_data


class rp_dataset(Dataset):
    """
    This class generates a pytorch dataset from given input directory
    """

    def __init__(self, mapping, data_dir, transform=None):
        self.data = data_dir  # directory for training data
        self.mapping = pd.read_csv(
            mapping, header=None, index_col=0
        )  # file for rp plot names
        self.transform = transform  # transformations

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        rp_path = os.path.join(
            self.data, self.mapping.iloc[idx, 0]
        )  # create path to plot
        rp = read_image(rp_path)
        y_label = torch.tensor(int(self.mapping.iloc[idx, 3]))

        if self.transform:
            rp = self.transform(rp)

        return rp, y_label


class GoogLeNet(nn.Module):
    """
    Initialises a modified version of pretrained GoogLeNet model.
    Returns out of dropout layer as a tool for feature extraction.
    """

    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.googlenet(
            weights="GoogLeNet_Weights.IMAGENET1K_V1"
        )
        self.layers = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        x = self.layers(x)
        return x

    def feature_extraction(self, dataloader):
        self.layers.eval()
        self.layers.to(self.device)

        results = None

        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                out = self.layers(X)
                if results is None:
                    results = np.asarray(torch.Tensor.cpu(out))
                else:
                    results = np.append(
                        results, np.asarray(torch.Tensor.cpu(out)), axis=0
                    )

        return results.squeeze()

    def summary(self):
        return torchinfo.summary(
            self.layers,
            (3, 224, 224),
            batch_dim=0,
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "mult_adds",
            ),
            verbose=0,
        )


class GoogLeNet_ft(nn.Module):
    """
    Initialises a googlenet model to be finetuned.
    Returns out of dropout layer as a tool for feature extraction.
    """

    def __init__(self):
        super(GoogLeNet_ft, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.googlenet(weights="GoogLeNet_Weights.DEFAULT")

        # specify new fc layer
        self.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2, bias=True),
            # nn.Sigmoid(),
        )

        # add fc to model layers
        self.layers = nn.Sequential(*list(self.model.children())[:-1], self.fc)

        # send to gpu
        self.layers.to(self.device)

        # freeze all layers and selectively unfreeze last inception layer for training
        for param in self.layers.parameters():
            param.requires_grad = False
        for param in self.layers[-1].parameters():
            param.requires_grad = True
        for param in self.layers[-2].parameters():
            param.requires_grad = True
        for param in self.layers[-3].parameters():
            param.requires_grad = True
        for param in self.layers[-4].parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.layers(x)

        return x

    def model_train(self, dataloader, optimiser, loss_fn, batch_size):
        self.layers.train()

        size = len(dataloader.dataset)

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            optimiser.zero_grad()

            pred = self.layers(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimiser.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # extracts features form the dropout layer of the NN
    def feature_extraction(self, dataloader):
        self.layers.eval()
        self.layers.to(self.device)

        results = None

        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                out = self.layers(X)
                if results is None:
                    results = np.asarray(torch.Tensor.cpu(out))
                else:
                    results = np.append(
                        results, np.asarray(torch.Tensor.cpu(out)), axis=0
                    )

        return results.squeeze()

    def summary(self):
        return torchinfo.summary(
            self.layers,
            (3, 224, 224),
            batch_dim=0,
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "mult_adds",
            ),
            verbose=0,
            depth=10,
        )
