# John Bachman Kelley, 10/19/2020, Artificial Intelligence
#   The purpose of this code is to predict energy consumption using three different architectures of linear regression
#   neurons. We have been given 3 days of data, from 8:00 to 5:00 pm with 1-hour intervals.
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class Neuron1D:

    # In this method, you need to import, normalize, and set object variables
    def __init__(self, csv_path, max_epoch, alpha):
        self.max_epoch = max_epoch
        self.TE = 0
        self.alpha = alpha
        self.raw_data = raw_data = np.genfromtxt(csv_path, delimiter=",")
        self.x = x = raw_data[:, 0]
        self.x = (self.x - min(x)) / (max(x) - min(x))
        self.real_y = raw_data[:, 1]
        self.y = np.zeros(x.shape[0])
        self.weights = np.ones(2)
        self.X = np.vstack([np.ones(x.shape[0]), self.x])
        return

    def train(self):
        for i in range(self.max_epoch):
            # Assess the Total Error of Changing
            self.predict()
            sigma = (self.real_y - self.y)
            for j in range(self.X.shape[1]):
                # We consider each feature to's value to adjust the weight
                # sigma[i] = error for that data point
                # alpha = learning constant
                # self.X[k][i] = feature k's prediction value
                self.weights += self.alpha * sigma[j] * self.X[0][j]
                self.weights[1] += self.alpha * sigma[j] * self.X[1][j]
        return

    def test(self):
        # Predict
        self.predict()
        self.assess_error()
        # print(f"Total Error: {self.TE} \nWeights: {self.weights[0]} {self.weights[1]}\n\n")
        return

    def predict(self):
        for i in range(self.x.shape[0]):
            # y = mx + b
            self.y[i] = (self.weights[1] * self.X[1][i]) + self.weights[0]
        return

    def plot(self):
        # Plot Results - Still needs the prediction line
        plt.scatter(self.x, self.y, c='g')
        plt.scatter(self.x, self.real_y, c='b')
        plt.show()
        return

    def assess_error(self):
        totalError = 0
        # Assess T.E
        for i in range(self.X.shape[1]):
            totalError += (self.y[i] - self.real_y[i]) ** 2
        self.TE = totalError
        return


class Neuron2D:

    # In this method, you need to import, normalize, and set object variables
    def __init__(self, csv_path, max_epoch, alpha):
        self.max_epoch = max_epoch
        self.TE = 0
        self.alpha = alpha
        self.raw_data = raw_data = np.genfromtxt(csv_path, delimiter=",")
        self.x = x = raw_data[:, 0]
        self.x = (self.x - min(x)) / (max(x) - min(x))
        self.real_y = raw_data[:, 1]
        self.y = np.zeros(x.shape[0])
        self.weights = np.ones(3)
        self.X = np.vstack([np.ones(x.shape[0]), self.x, self.x ** 2])
        return

    def train(self):
        for i in range(self.max_epoch):
            # Assess the Total Error of Changing
            self.predict()
            sigma = (self.real_y - self.y)
            for j in range(self.X.shape[1]):
                # We consider each feature to's value to adjust the weight
                # sigma[i] = error for that data point
                # alpha = learning constant
                # self.X[k][i] = feature k's prediction value
                self.weights += self.alpha * sigma[j] * self.X[0][j]
                self.weights[1] += self.alpha * sigma[j] * self.X[1][j]
                self.weights[2] += self.alpha * sigma[j] * self.X[2][j]
        return

    def test(self):
        # Predict
        self.predict()
        self.assess_error()
        # print(f"Total Error: {self.TE} \nWeights: {self.weights[0]} {self.weights[1]}\n\n")
        return

    def predict(self):
        for i in range(self.x.shape[0]):
            # y = mx + b
            self.y[i] = (self.weights[2] * self.X[2][i]) + (self.weights[1] * self.X[1][i]) + self.weights[0]
        return

    def plot(self):
        # Plot Results - Still needs the prediction line
        plt.scatter(self.x, self.y, c='g')
        plt.scatter(self.x, self.real_y, c='b')
        plt.show()
        return

    def assess_error(self):
        totalError = 0
        # Assess T.E
        for i in range(self.X.shape[1]):
            totalError += (self.y[i] - self.real_y[i]) ** 2
        self.TE = totalError
        return


class Neuron3D:

    # In this method, you need to import, normalize, and set object variables
    def __init__(self, csv_path, max_epoch, alpha):
        self.max_epoch = max_epoch
        self.TE = 0
        self.alpha = alpha
        self.raw_data = raw_data = np.genfromtxt(csv_path, delimiter=",")
        self.x = x = raw_data[:, 0]
        self.x = (self.x - min(x)) / (max(x) - min(x))
        self.real_y = raw_data[:, 1]
        self.y = np.zeros(x.shape[0])
        self.weights = np.ones(4)
        self.X = np.vstack([np.ones(x.shape[0]), self.x, self.x ** 2, self.x ** 3])
        return

    def train(self):
        for i in range(self.max_epoch):
            # Assess the Total Error of Changing
            self.predict()
            sigma = (self.real_y - self.y)
            for j in range(self.X.shape[1]):
                # We consider each feature to's value to adjust the weight
                # sigma[i] = error for that data point
                # alpha = learning constant
                # self.X[k][i] = feature k's prediction value
                self.weights += self.alpha * sigma[j] * self.X[0][j]
                self.weights[1] += self.alpha * sigma[j] * self.X[1][j]
                self.weights[2] += self.alpha * sigma[j] * self.X[2][j]
                self.weights[3] += self.alpha * sigma[j] * self.X[3][j]
        return

    def test(self):
        # Predict
        self.predict()
        self.assess_error()
        # print(f"Total Error: {self.TE} \nWeights: {self.weights[0]} {self.weights[1]}\n\n")
        return

    def predict(self):
        for i in range(self.x.shape[0]):
            # y = mx + b
            self.y[i] = (self.weights[3] * self.X[3][i]) + (self.weights[2] * self.X[2][i]) + (
                        self.weights[1] * self.X[1][i]) + self.weights[0]
        return

    def plot(self):
        # Plot Results - Still needs the prediction line
        plt.scatter(self.x, self.y, c='g')
        plt.scatter(self.x, self.real_y, c='b')
        plt.show()
        return

    def assess_error(self):
        totalError = 0
        # Assess T.E
        for i in range(self.X.shape[1]):
            totalError += (self.y[i] - self.real_y[i]) ** 2
        self.TE = totalError
        return


# Main Script #

# Grid Search to Identify Best Alpha Values

# Create the Testing Grid
alpha_arr = np.arange(start=0.01, stop=0.001, step=-0.0005)
epoch_arr = np.arange(start=500, stop=8000, step=500)
best_alpha = 0.00001
best_epoch = 10000


class Found(Exception):
    """Exception for when Convergence is determined"""
    pass


# Run Tuning for Architecture 1
try:
    min_TE = 500
    best_alpha = 0
    best_epoch = 0
    delta_E = 5
    for alpha_index in range(alpha_arr.shape[0]):
        for epoch_index in range(epoch_arr.shape[0]):
            obj1 = Neuron1D('training_data.txt', epoch_arr[epoch_index], alpha_arr[alpha_index])
            obj1.train()
            obj1.test()
            if obj1.TE < min_TE:
                delta_E = min_TE - obj1.TE
                min_TE = obj1.TE
                best_alpha = obj1.alpha
                best_epoch = obj1.max_epoch
                if delta_E < 0.01:
                    raise Found
    print(f"Architecture1:\n    Best Alpha: {best_alpha}\n    Best Epoch: {best_epoch}")
    obj1.plot()
except Found:
    print(f"Architecture1:\n    Best Alpha: {best_alpha}\n    Best Epoch: {best_epoch}")
    obj1.plot()

# Run Tuning for Architecture 2
try:
    min_TE = 500
    best_alpha = 0
    best_epoch = 0
    delta_E = 5
    for alpha_index in range(alpha_arr.shape[0]):
        for epoch_index in range(epoch_arr.shape[0]):
            obj2 = Neuron2D('training_data.txt', epoch_arr[epoch_index], alpha_arr[alpha_index])
            obj2.train()
            obj2.test()
            if obj2.TE < min_TE:
                delta_E = min_TE - obj2.TE
                min_TE = obj2.TE
                best_alpha = obj2.alpha
                best_epoch = obj2.max_epoch
                if delta_E < 0.01:
                    raise Found
    print(f"Architecture2:\n    Best Alpha: {best_alpha}\n    Best Epoch: {best_epoch}")
    obj2.plot()
except Found:
    print(f"Architecture2:\n    Best Alpha: {best_alpha}\n    Best Epoch: {best_epoch}")
    obj2.plot()

# Run Tuning for Architecture 3
try:
    min_TE = 500
    best_alpha = 0
    best_epoch = 0
    delta_E = 5
    for alpha_index in range(alpha_arr.shape[0]):
        for epoch_index in range(epoch_arr.shape[0]):
            obj3 = Neuron3D('training_data.txt', epoch_arr[epoch_index], alpha_arr[alpha_index])
            obj3.train()
            obj3.test()
            if obj3.TE < min_TE:
                delta_E = min_TE - obj3.TE
                min_TE = obj3.TE
                best_alpha = obj3.alpha
                best_epoch = obj3.max_epoch
                if delta_E < 0.01:
                    raise Found
    print(f"Architecture3:\n    Best Alpha: {best_alpha}\n    Best Epoch: {best_epoch}")
    obj3.plot()
except Found:
    print(f"Architecture3:\n    Best Alpha: {best_alpha}\n    Best Epoch: {best_epoch}")
    obj3.plot()
