from __future__ import print_function

import numpy as np
import argparse

import sys
import os
# import glob
import mayavi.mlab as mlab
# from utils.visualization_utils import *
import ipdb


mlab.figure()
# t = np.linspace(0, 20, 200)
# mlab.plot3d(np.sin(t), np.cos(t), 0.1*t, t)

mlab.show()
ipdb.set_trace()


def prova():
  t = np.linspace(0, 20, 200)
  mlab.plot3d(np.sin(t), np.cos(t), 0.1*t, t)
  print("Hello, World!")
