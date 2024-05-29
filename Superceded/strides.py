import numpy as np
import math
from numpy.lib.stride_tricks import as_strided


def sliding_window_slicing(a, no_items, item_type=0):
    """This method perfoms sliding window slicing of numpy arrays
    Credit to https://stackoverflow.com/questions/53097952/how-to-understand-numpy-strides-for-layman. User Nikola V for the implementation

    Parameters
    ----------
    a : numpy
        An array to be slided in subarrays
    no_items : int
        Number of sliced arrays or elements in sliced arrays
    item_type: int
        Indicates if no_items is number of sliced arrays (item_type=0) or
        number of elements in sliced array (item_type=1), by default 0

    Return
    ------
    numpy
        Sliced numpy array
    """
    if item_type == 0:
        no_slices = no_items
        no_elements = len(a) + 1 - no_slices
        if no_elements <= 0:
            raise ValueError(
                "Sliding slicing not possible, no_items is larger than " + str(len(a))
            )

    else:
        no_elements = no_items
        no_slices = len(a) - no_elements + 1
        if no_slices <= 0:
            raise ValueError(
                "Sliding slicing not possible, no_items is larger than " + str(len(a))
            )

    subarray_shape = a.shape[1:]
    shape_cfg = (no_slices, no_elements) + subarray_shape
    strides_cfg = (a.strides[0],) + a.strides
    as_strided = np.lib.stride_tricks.as_strided  # shorthand
    return as_strided(a, shape=shape_cfg, strides=strides_cfg)


def window(a, w=4, o=2, copy=False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view


def main():
    """
    cols = 30851
    cols = cols
    rows = 23
    length = 200
    windows = 10
    a = np.arange(cols * rows).reshape(rows, cols)  # (23, 30851)
    a = np.asfarray(a)
    a = a.T  # (30851, 23)
    """

    step_size = 10
    length = 5

    a = np.arange(100).reshape(5, 20)
    a = np.asfarray(a)
    # a = a.T

    stride = a.strides

    # print(a.shape)
    # print(stride)
    # print(a.shape[1:])
    # print(a.size)

    # required to move 8 bytes to move 1 element
    # therefore we must multiply the number of elements in a row by 8 to traverse the entire row
    # e.g. shape of (5,20) == 20 * 8 = 160 per row.
    # stride for this shape is (160,8)

    # therefore, if I want to create a new array that has 10 elements (window) per row, I need to traverse half the row (in bytes == 80), and then create a new row with this data, and so on.

    # works for 1d
    # ee = a.strides * 2
    # print(ee)

    # print(a)
    # works for 2d
    # bb = (a.strides[0],) + a.strides
    # print(bb)
    """
    item_type: int
        Indicates if no_items is number of sliced arrays (item_type=0) or
        number of elements in sliced array (item_type=1), by default 0
    """
    # for shape
    print(a.strides[1])
    # subarray_shape = a.shape[1:]
    # print(a.shape[1:])

    # print(len(a))

    # no_slices = len(a) - no_elements + 1
    aa = 20 - length + 1
    print(f"aa is {aa}")

    # shape
    # shape_cfg = (no_slices, no_elements) + subarray_shape
    bb = (aa, length) + a.shape[1:]
    # bb = a.shape[1:] + (aa, length)
    print(f"bb is {bb}")

    # strides
    # strides_cfg = (a.strides[0],) + a.strides
    # cc = (a.strides[0],) + a.strides
    cc = (a.strides[1], a.strides[1] * step_size, a.strides[1])
    print(f"cc is {cc}")

    # shape = (a.size - length + 1, length) + a.shape[1:]

    # print(shape)

    test = as_strided(a, shape=bb, strides=cc)

    # print(test.T)

    print(test.shape)
    print(test[15])
    # print(test[9])


if __name__ == "__main__":
    main()
