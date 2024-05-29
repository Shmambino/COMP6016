## Function Declarations for Campbell Eaton [18824421] - COMP6016 - Final Project

import pandas as pd
import numpy as np
import polars as pl
import polars.selectors as cs
import math
from pyrqa.opencl import OpenCL
from pathlib import Path

from sklearn.preprocessing import StandardScaler

from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator
from pyrqa.analysis_type import Classic
from pyrqa.time_series import TimeSeries

import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision
from torchvision.transforms import v2  # Transformations we can perform on our dataset
from torchvision.io import read_image
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np


"Pandas Functions"


def check_conversion(dataframe: pd.DataFrame) -> bool:
    """
    Checking to see if conversion was successful, returning 0 na's.
    """

    if dataframe["Value"].isna().sum() == 0:
        return True


def pivot(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot dataframe to long format.
    """

    dataframe = dataframe.pivot_table(index="TimeStamp", columns="Name", values="Value")

    return dataframe


def resample_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Resample dataframe with basic interpolation of missing values
    """

    dataframe.reset_index(inplace=True)
    dataframe["TimeStamp"] = pd.to_datetime(dataframe["TimeStamp"])
    dataframe.set_index("TimeStamp", inplace=True)

    for column in dataframe.columns:
        dataframe[column] = dataframe[column].ffill()
        dataframe[column] = dataframe[column].bfill()

    dataframe = dataframe.resample("5s").first()
    dataframe.dropna(how="all", inplace=True)

    return dataframe


"*********************************************"

"Polars Functions - ideal to use as polars is much more performant than pandas."


def check_conversion_pl(dataframe: pl.DataFrame) -> bool:
    """
    Checking to see if conversion was successful, returning 0 na's.
    """

    if dataframe.select(pl.sum("Value")).null_count().collect().item() == 0:

        return True


def pivot_pl(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Pivot dataframe to long format.
    """

    result = dataframe.collect().pivot(
        index="TimeStamp", columns="Name", values="Value", aggregate_function="first"
    )

    return result.lazy()


def resample_pl(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Resample dataframe with basic interpolation of missing values
    """

    dataframe = dataframe.select(pl.all().forward_fill())
    dataframe = dataframe.select(pl.all().backward_fill())

    dataframe = (
        dataframe.sort(by="TimeStamp")
        .group_by_dynamic(index_column="TimeStamp", every="5s", check_sorted=False)
        .agg(pl.exclude("TimeStamp").first())
    )

    return dataframe.lazy()


def drop_pl(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Drops columns
    """

    dataframe = dataframe.drop(
        ~cs.contains(
            ("Temperature", "Pressure", "Engine", "Transmission", "Speed", "Time")
        )
    )

    dataframe = dataframe.drop(
        cs.contains(
            (
                "Total",
                "Transmission Hours",
                "Ambient",
                "Engine Hours",
                "Tire Pressure",
                "Catalyst",
                "Engine Running",
                "Transmission Retarder Active",
                "Instant Fuel",
                "DPF Pressure",
                "DPF Temperature",
                "Upbox Oil Temperature",
            )
        )
    ).unique(maintain_order=True)

    return dataframe.lazy()


def save_rqa_features(result_obj, dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Add results from RQA to dataframe for future analysis.
    """

    # retrieve new data
    data = result_obj.to_array()
    data = np.reshape(data, (1, data.shape[0]))

    # create new dataframe
    new_df = pl.DataFrame(
        data=data,
        schema={
            "MDL",
            "MVL",
            "MWVL",
            "RR",
            "DET",
            "ADL",
            "LDL",
            "DIV",
            "EDL",
            "LAM",
            "TT",
            "LVL",
            "EVL",
            "AWVL",
            "LWVL",
            "LWVLI",
            "EWVL",
            "Ratio_DRR",
            "Ratio_LD",
        },
        orient="row",
    )

    # append new dataframe to results
    dataframe = dataframe.concat(new_df).lazy()

    return dataframe


"*********************************************"

"Reccurence Plot Functions"


def reshape_input(dataframe: pl.DataFrame) -> np.array:
    """
    Convert dataframe to array and reshape.
    """

    result = dataframe.collect().to_numpy()
    result = np.swapaxes(result, 0, 1)

    return result


def scale_input(a: np.array) -> np.array:
    """
    Scale input array with standard scaler.
    """

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    result = scaler.fit_transform(a)

    return result


def window(a, w=4, o=4, copy=False):
    s0, s1 = a.strides
    m, n = a.shape
    view = np.lib.stride_tricks.as_strided(
        a, strides=(s1, s0, s1), shape=(n - w + 1, m, w)
    )[0::o]
    if copy:
        return view.copy()
    else:
        return view


def plot_reccurence(
    input_array: np.array,
    stride: int,
    iteration: int,
    dates: pl.DataFrame,
    data_idx: int,
    distance: int,
    window: int,
    target: str,
):
    """
    Generates reccurance plots and RQA matrix for input array.

    Slices input array based on user specified parameter stide.

    Saves plots and RQA matrix for further analysis, as well as a mapping file of plots to data index, sensor names, and date range of the sensor for future reference.

    """
    if window == None:
        opencl = OpenCL(platform_id=0, device_ids=(0,))
        splits = math.floor(input_array.shape[1] / stride)  # generate number of splits
        array_list = np.array_split(input_array, splits, axis=1)  # split array
    else:
        pass

    first = True
    rqa_df = None
    columns = [
        "MDL",
        "MVL",
        "MWVL",
        "RR",
        "DET",
        "ADL",
        "LDL",
        "DIV",
        "EDL",
        "LAM",
        "TT",
        "LVL",
        "EVL",
        "AWVL",
        "LWVL",
        "LWVLI",
        "EWVL",
        "Ratio_DRR",
        "Ratio_LD",
    ]
    start_date_idx = 0
    end_date_idx = 0

    if target == "Healthy":
        direct = "Train-Healthy"
        label = 1
    if target == "Unhealthy":
        direct = "Train-Unhealthy"
        label = 0

    for i, a in enumerate(array_list):  # for each array in arraylist
        for j, _ in enumerate(a):

            # Generate RQA Matrix
            ts = TimeSeries(a[j], embedding_dimension=3, time_delay=1)

            rp = Settings(
                ts,
                analysis_type=Classic,
                neighbourhood=FixedRadius(distance),
                similarity_measure=EuclideanMetric,
                theiler_corrector=1,
            )

            computation = RQAComputation.create(rp, opencl=opencl, verbose=False)

            result = computation.run()

            result.min_diagonal_line_length = 2
            result.min_vertical_line_length = 2
            result.min_white_vertical_line_length = 2

            # add results from run to df
            if first == True:
                result = result.to_array()
                result = np.reshape(result, (1, result.shape[0]))
                rqa_df = pl.DataFrame(data=result, schema=columns, orient="row").lazy()

            else:
                result = result.to_array()
                result = np.reshape(result, (1, result.shape[0]))
                new_results = pl.DataFrame(
                    data=result, schema=columns, orient="row"
                ).lazy()
                rqa_df = pl.concat([rqa_df, new_results]).lazy()

            # filepath
            path = Path(f"../RP/{direct}/distance_{distance}/stride_{stride}/").mkdir(
                parents=True, exist_ok=True
            )

            # generate RP
            computation = RPComputation.create(rp, opencl=opencl, verbose=False)
            result = computation.run()
            ImageGenerator.save_recurrence_plot(
                result.recurrence_matrix_reverse,
                f"../RP/{direct}/distance_{distance}/stride_{stride}/Iteration_{iteration}_Split_{i}_Sensor_{j}_{target}.png",
            )

            # get start and end date of array for this split
            end_date_idx = start_date_idx + a.shape[1]
            start_date = dates.slice(start_date_idx, 1).collect().item()
            end_date = dates.slice(end_date_idx - 1, 1).collect().item()

            # save rp filenames with time window, sensor name, and label (in this case, 0 = normal)
            with open(
                f"../RP/{direct}/distance_{distance}/stride_{stride}/mapping.csv",
                "a",
            ) as fileobj:
                fileobj.write(
                    f"{data_idx},Iteration_{iteration}_Split_{i}_Sensor_{j}_{target}.png,{start_date},{end_date},{label}\n"
                )
                data_idx += 1

            # set flag to false for df generation
            first = False

        start_date_idx = end_date_idx

    return rqa_df, data_idx
