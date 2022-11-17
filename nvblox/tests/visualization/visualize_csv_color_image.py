#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as pyplot
import sys


def main(argv):

    color_frame = np.genfromtxt(argv[1], dtype=int)

    num_rows = color_frame.shape[0]
    num_cols = color_frame.size / (num_rows * 3)

    print(num_cols)

    color_frame = color_frame.reshape((num_rows, -1, 4))

    print(color_frame.shape)

    single_channel = np.squeeze(color_frame[:,:,0:4])

    print(single_channel.dtype)

    pyplot.imshow(color_frame)
    pyplot.show()


if __name__ == "__main__":
    main(sys.argv)
