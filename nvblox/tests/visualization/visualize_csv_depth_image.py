#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as pyplot
import sys


def main(argv):

    depth_frame = np.genfromtxt(argv[1])

    pyplot.imshow(depth_frame)
    pyplot.colorbar()
    pyplot.show()


if __name__ == "__main__":
    main(sys.argv)
