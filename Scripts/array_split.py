import numpy as np
import math


def main():
    n_rows = 23
    n_columns = 30851
    length = 200
    test = np.arange(n_rows * n_columns).reshape(n_rows, n_columns)

    print(f"Original shape = {test.shape}")

    splits = math.floor(test.shape[1] / length)  # generate number of splits
    array_list = np.array_split(test, splits, axis=1)  # split array

    print(
        f"Length of array list = {len(array_list)} arrays"
    )  # x arrays, each corresponding to a different time

    # print(array_list[0].shape)  # with each row corresponding to a sensor
    print(f"Minimum array dimensions are {array_list[-1].shape}")


#
# print(array_list[0][1])
# print(array_list[1][0])


if __name__ == "__main__":
    main()
