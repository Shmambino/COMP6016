## Data Processing script for Campbell Eaton [18824421] - COMP6016 - Final Project
import polars as pl
from Scripts.classes_z import dataset
from pathlib import Path


def main():

    distance = [0.025]  # distance threshold
    stride = [30]  # stride across timeseries (lower values mean more overlap)
    length = [300]  # sliding window length

    for d in distance:
        for s in stride:
            for w in length:

                # read in raw data
                healthy_raw_df = pl.scan_parquet(
                    "../Data/conditionmeasurements_0277_no_breakdown.parquet/*.parquet",
                    n_rows=18000000,
                )

                # initialise dataset classes
                healthy_dataset = dataset(
                    raw_data=healthy_raw_df,
                    n_rows=1000000,
                    stride=s,
                    iterations=18,  # depends on n_rows and shape of raw data
                    distance=d,
                    length=w,
                    target="Healthy",
                )

                # perform generaiton of recurrence plots and RQA feature matrix's
                print("**Performing transforms and generating data..")
                healthy_dataset.transform()
                print("**Generation complete\n")

                # return feature dataframes
                healthy_features_df = healthy_dataset.out_features()

                # save feature dataframes
                healthy_features_df.collect().write_csv(
                    f"../Data/healthy_rp_distance{d}_length_{w}_stride{s}_2.csv"
                )


if __name__ == "__main__":
    main()
