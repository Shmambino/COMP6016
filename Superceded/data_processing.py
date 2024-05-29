## Data Processing script for Campbell Eaton [18824421] - COMP6016 - Final Project
import polars as pl
from Scripts.classes_z import dataset
from pathlib import Path


def main():

    distance = [0.01, 0.025, 0.05]  # distance threshold
    stride = [10, 20, 30]  # stride across timeseries (lower values mean more overlap)
    window = [100, 200, 300]  # sliding window length

    for d in distance:
        for s in stride:
            for w in window:

                # read in raw data
                healthy_raw_df = pl.scan_parquet(
                    "../Data/conditionmeasurements_0277_no_breakdown.parquet/*.parquet",
                    n_rows=18000000,
                )
                unhealthy_raw_df = pl.scan_parquet(
                    "../Data/conditionnmeasurements_0277_20230104_20230107.parquet/*.parquet",
                    n_rows=1000000,
                )

                # initialise dataset classes
                healthy_dataset = dataset(
                    raw_data=healthy_raw_df,
                    n_rows=1000000,
                    stride=s,
                    iterations=18,  # depends on n_rows and shape of raw data
                    distance=d,
                    window=w,
                    target="Healthy",
                )
                unhealthy_dataset = dataset(
                    raw_data=unhealthy_raw_df,
                    n_rows=1000000,
                    stride=s,
                    iterations=1,  # depends on n_rows and shape of raw data
                    distance=d,
                    window=w,
                    target="Unhealthy",
                )

                # perform generaiton of recurrence plots and RQA feature matrix's
                print("**Performing transforms and generating data..")
                healthy_dataset.transform()
                print("**Generation complete\n")

                print("**Performing transforms and generating data..")
                unhealthy_dataset.transform()
                print("**Generation complete\n")

                # return feature dataframes
                healthy_features_df = healthy_dataset.out_features()
                unhealthy_features_df = unhealthy_dataset.out_features()

                # save feature dataframes
                healthy_features_df.collect().write_csv(
                    f"../Data/healthy_rp_distance{d}_stride{s}.csv"
                )
                unhealthy_features_df.collect().write_csv(
                    f"../Data/unhealthy_rp_distance{d}_stride{s}.csv"
                )


if __name__ == "__main__":
    main()
