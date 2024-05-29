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
from torcheval.metrics.functional import binary_accuracy


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
        distance: float,
        length: int,
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
        self.distance = distance
        self.length = length

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
            self.distance,
            self.length,
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
            mapping, header=None, index_col=None
        )  # file for rp plot names
        self.transform = transform  # transformations

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        rp_path = os.path.join(
            self.data, self.mapping.iloc[idx, 1]
        )  # create path to plot
        rp = read_image(rp_path)
        y_label = torch.tensor(int(self.mapping.iloc[idx, 4]))

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
        self.model = torchvision.models.googlenet(
            weights="GoogLeNet_Weights.DEFAULT", aux_logits=True
        )
        self.train_loss = []
        self.test_loss = []

        # define aux classifiers (dropout taken from model definition)
        self.model.aux1 = self.model.inception_aux_block(512, 1, dropout=0.7)
        self.model.aux2 = self.model.inception_aux_block(528, 1, dropout=0.7)

        # specify new fc layer
        self.model.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1, bias=True),
            # nn.Sigmoid(),
        )

        self.model.to(self.device)

        # freeze all layers and selectively unfreeze last inception layer for training
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.avgpool.parameters():
            param.requires_grad = True
        for param in self.model.dropout.parameters():
            param.requires_grad = True
        for param in self.model.aux1.parameters():
            param.requires_grad = True
        for param in self.model.aux2.parameters():
            param.requires_grad = True
        """
        for param in self.model.inception5b.parameters():
            param.requires_grad = 
        """

    def forward(self, x):
        x = self.model(x)
        return x

    def model_train(self, dataloader1, dataloader2, optimiser, loss_fn, batch_size):
        self.model.train()

        size = len(dataloader1.dataset)
        num_batches = len(dataloader1)
        losses = []

        for batch, (data1, data2) in enumerate(zip(dataloader1, dataloader2)):

            X1, y1 = data1
            X2, y2 = data2

            # get first dataloader batch
            X1, y1 = X1.to(self.device), y1.float().to(self.device)

            pred, aux1, aux2 = self.model(X1)
            y1 = y1.unsqueeze(1)

            loss_b1 = loss_fn(pred, y1)
            loss_b1_aux1 = loss_fn(aux1, y1)
            loss2_b1_aux2 = loss_fn(aux2, y1)

            loss_b1 = loss_b1 * 0.3 * (loss_b1_aux1 + loss2_b1_aux2)

            # get second dataloader batch
            X2, y2 = X2.to(self.device), y2.float().to(self.device)

            pred, aux1, aux2 = self.model(X2)
            y2 = y2.unsqueeze(1)

            loss_b2 = loss_fn(pred, y2)
            loss_b2_aux1 = loss_fn(aux1, y2)
            loss_b2_aux2 = loss_fn(aux2, y2)

            loss_b2 = loss_b2 * 0.3 * (loss_b1_aux1 + loss_b2_aux2)

            # sum losses over batches
            loss = loss_b1 + loss_b2

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X1)
                print(f"loss: {loss:>7f}  [{current}/{size}]")

        self.train_loss.append(sum(losses) / num_batches)

    def model_test(self, dataloader1, dataloader2, loss_fn):
        self.model.eval()

        size = len(dataloader1.dataset) * 2

        num_batches = len(dataloader1) * 2
        test_loss, correct = 0, 0

        with torch.no_grad():
            for batch, (data1, data2) in enumerate(zip(dataloader1, dataloader2)):
                X1, y1 = data1
                X2, y2 = data2

                X1, y1 = X1.to(self.device), y1.float().to(self.device)
                pred = self.model(X1)
                y1 = y1.unsqueeze(1)
                test_loss += loss_fn(pred, y1).item()
                correct += self.correct(pred, y1)

                X2, y2 = X2.to(self.device), y2.float().to(self.device)
                pred = self.model(X2)
                y2 = y2.unsqueeze(1)
                test_loss += loss_fn(pred, y2).item()
                correct += self.correct(pred, y2)

        test_loss /= num_batches
        correct /= size
        self.test_loss.append(test_loss)
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    def correct(self, predictions, labels):
        x = torch.sigmoid(predictions)
        x = torch.where(x < 0.50, 0, 1)

        correct = (x == labels).float().sum()
        return correct

    # extracts features form the dropout layer of the NN
    def feature_extraction(self, dataloader):
        self.model.eval()
        self.model.to(self.device)

        results = None

        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.float().to(self.device)
                out = self.model(X)
                if results is None:
                    results = np.asarray(torch.Tensor.cpu(out))
                else:
                    results = np.append(
                        results, np.asarray(torch.Tensor.cpu(out)), axis=0
                    )

        return results.squeeze()

    def summary(self):
        return torchinfo.summary(
            self.model,
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
