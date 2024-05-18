## Data Processing script for Campbell Eaton [18824421] - COMP6016 - Final Project
import polars as pl
from classes import dataset
from pathlib import Path


def main():

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
        stride=200,
        iterations=18,  # depends on n_rows and shape of raw data
        target="normal",
    )
    unhealthy_dataset = dataset(
        raw_data=unhealthy_raw_df,
        n_rows=1000000,
        stride=200,
        iterations=1,  # depends on n_rows and shape of raw data
        target="abnormal",
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

    # specify filenames to save
    output_file1 = "healthy_rp.csv"
    output_file2 = "unhealthy_rp.csv"

    # specify output directories
    out_dir1 = Path("../Data")
    out_dir2 = Path("../Data")
    out_dir1.mkdir(parents=True, exist_ok=True)
    out_dir2.mkdir(parents=True, exist_ok=True)

    # save feature dataframes
    healthy_features_df.collect().write_csv(output_file1 / out_dir1)
    unhealthy_features_df.collect().write_csv(output_file1 / out_dir1)


if __name__ == "__main__":
    main()
