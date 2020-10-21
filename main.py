# John Bachman Kelley, 10/19/2020, Artificial Intelligence
#   The purpose of this code is to predict energy consumption using three different architectures of linear regression
#   neurons. We have been given 3 days of data, from 8:00 to 5:00 pm with 1-hour intervals.
import pandas as pd
import numpy as np
import matplotlib as plt

class Neuron1D:

    # In this method, you need to import, normalize, and set object variables
    def __init__(self, csv_path):
        self.raw_data = raw_data = np.genfromtxt(csv_path, delimiter=",")
        self.x = x = raw_data[:, 0]
        self.real_y = y = raw_data[:, 1]
        self.y = np.empty(x.shape[0])
        self.weights = np.ones(2)
        self.ones = np.ones(x.shape[0])
        return

    def train(self):
        return

    def test(self):
        return

    def predict(self):
        for i in range(self.x):
            # y = mx + b
            self.y[i] = (self.weights[1] * self.x[i]) + self.weights[0] * self.ones[i]
        return


class Neuron2D:

    # In this method, you need to import, normalize, and set object variables
    def __init__(self, csv_path):
        raw_data = np.genfromtxt(csv_path, delimiter=",")
        x = raw_data[:, 0]
        y = raw_data[:, 1]
        ones = np.ones(x.shape[0])
        return

    def train(self):
        return

    def test(self):
        return

    def predict(self):
        return


class Neuron3D:

    # In this method, you need to import, normalize, and set object variables
    def __init__(self, csv_path):
        raw_data = np.genfromtxt(csv_path, delimiter=",")
        x = raw_data[:, 0]
        y = raw_data[:, 1]
        ones = np.ones(x.shape[0])
        return

    def train(self):
        return

    def test(self):
        return

    def predict(self):
        return

# Main Script #
obj1 = Neuron1D('test_data_4.txt')


