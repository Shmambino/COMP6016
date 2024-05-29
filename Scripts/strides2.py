import numpy as np
import math
from numpy.lib.stride_tricks import as_strided


def window(a, w=4, o=4, copy=False):
    sh = (a.size - w + 1, w, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view


def window2(a, w=4, o=4, copy=False):
    s0, s1 = a.strides
    m, n = a.shape
    view = np.lib.stride_tricks.as_strided(
        a, strides=(s1, s0, s1), shape=(n - w + 1, m, w)
    )[0::o]
    if copy:
        return view.copy()
    else:
        return view


def main():

    cols = 30851
    rows = 23
    length = 200
    stride = 20
    stride_perc = 1 - stride / length

    print(stride_perc)
    a = np.arange(cols * rows).reshape(rows, cols)
    a = np.asfarray(a)

    print(a.shape)  # (23, 30851)

    test = window2(a, length, stride, copy=True)

    # print(test.shape)  # (1533, 23, 200)
    """
    print(test[0, :, :].shape)  # (23, 200) #first matrix, all rows and columns
    print(test[0, :, :])  # first matrix, all rows and columns
    print("***************")
    print("First split")
    print(test[0, :, 0])  # first matrix, all rows, first column
    print("***************")
    print("Second split")
    print(test[1, :, 0])  # Second matrix, all rows, first column
    print("Second split")
    print(test[1, :, :])  # Second matrix, all rows, first column

    # print(test[0, :, 0])
    # print(test[1, :, 0].shape)
    """

    arr = test[0, :, :]

    print(arr.shape)
    # print(test[0, :, :].shape)

    # array_list = []

    # for i in range(test.shape[0]):
    #    print(i)

    # print(test.shape)


if __name__ == "__main__":
    main()
