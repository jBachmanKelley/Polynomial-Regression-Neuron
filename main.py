# John Bachman Kelley, 10/19/2020, Artificial Intelligence
#   The purpose of this code is to predict energy consumption using three different architectures of linear regression
#   neurons. We have been given 3 days of data, from 8:00 to 5:00 pm with 1-hour intervals.
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class Neuron1D:

    # In this method, you need to import, normalize, and set object variables
    def __init__(self, csv_path, max_epoch, epsilon, alpha):
        self.max_epoch = max_epoch
        self.TE = 0
        self.alpha = alpha
        self.epsilon = epsilon
        self.raw_data = raw_data = np.genfromtxt(csv_path, delimiter=",")
        self.x = x = raw_data[:, 0]
        self.x = (self.x - min(x)) / (max(x) - min(x))
        self.real_y = raw_data[:, 1]
        self.y = np.zeros(x.shape[0])
        self.weights = np.ones(2)
        self.X = np.vstack([np.ones(x.shape[0]), self.x])
        return

    def train(self):
        sigma = self.y
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
        self.TE = sum(sigma)
        return

    def test(self):
        # Predict
        self.predict()
        totalError = 0

        # Assess T.E
        for i in range(self.X.shape[1]):
            totalError += (self.y[i] - self.real_y[i])**2
        print(f"Total Error: {totalError} \nWeights: {self.weights[0]} {self.weights[1]}\n\n")
        return totalError

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


class Neuron2D:

    # In this method, you need to import, normalize, and set object variables
    def __init__(self, csv_path, max_epoch, epsilon, alpha):
        self.max_epoch = max_epoch
        self.alpha = alpha
        self.epsilon = epsilon
        self.raw_data = raw_data = np.genfromtxt(csv_path, delimiter=",")
        self.x = x = raw_data[:, 0]
        self.x = (self.x - min(x)) / (max(x) - min(x))
        self.real_y = raw_data[:, 1]
        self.y = np.zeros(x.shape[0])
        self.weights = np.ones(3)
        self.X = np.vstack([np.ones(x.shape[0]), self.x, self.x**2])
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
        totalError = 0

        # Assess T.E
        for i in range(self.X.shape[1]):
            totalError += (self.y[i] - self.real_y[i])**2
        print(f"Total Error: {totalError} \nWeights: {self.weights[0]} {self.weights[1]} {self.weights[2]}\n\n")
        # Plot Results - Still needs the prediction line
        plt.scatter(self.x, self.y, c='g')
        plt.scatter(self.x, self.real_y, c='b')
        plt.show()
        return totalError

    def predict(self):
        for i in range(self.x.shape[0]):
            # y = mx + b
            self.y[i] = (self.weights[2] * self.X[2][i]) + (self.weights[1] * self.X[1][i]) + self.weights[0]
        return



class Neuron3D:

    # In this method, you need to import, normalize, and set object variables
    def __init__(self, csv_path, max_epoch, epsilon, alpha):
        self.max_epoch = max_epoch
        self.alpha = alpha
        self.epsilon = epsilon
        self.raw_data = raw_data = np.genfromtxt(csv_path, delimiter=",")
        self.x = x = raw_data[:, 0]
        self.x = (self.x - min(self.x)) / (max(self.x) - min(self.x))
        self.real_y = raw_data[:, 1]
        self.y = np.zeros(x.shape[0])
        self.weights = np.ones(4)
        self.X = np.vstack([np.ones(x.shape[0]), self.x, self.x**2, self.x**3])
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
                # sign = the sign of the change in sigma

                self.weights += self.alpha * sigma[j] * self.X[0][j]
                self.weights[1] += self.alpha * sigma[j] * self.X[1][j]
                self.weights[2] += self.alpha * sigma[j] * self.X[2][j]
                self.weights[3] += self.alpha * sigma[j] * self.X[3][j]
        return

    def test(self):
        # Predict
        self.predict()
        totalError = 0

        # Assess T.E
        for i in range(self.X.shape[1]):
            totalError += (self.y[i] - self.real_y[i])**2
        print(f"Total Error: {totalError} \nWeights: {self.weights[0]} {self.weights[1]} {self.weights[2]} {self.weights[3]}\n\n")
        # Plot Results - Still needs the prediction line
        plt.scatter(self.x, self.y, c='g')
        plt.scatter(self.x, self.real_y, c='b')
        plt.show()
        return totalError

    def predict(self):
        for i in range(self.X.shape[1]):
            # y = mx + b
            self.y[i] = (self.weights[3] * self.X[3][i]) + (self.weights[2] * self.X[2][i]) + (self.weights[1] * self.X[1][i]) + self.weights[0]
        return


# Main Script #

# Grid Search to Identify Best Alpha Values

# Create the Testing Grid
alpha_arr = np.arange(start=0.01, stop=0.001, step=-0.0005)
epoch_arr = np.arange(start=500, stop=5000, step=500)
min_TE = 500
best_alpha = 0
best_epoch = 0

for alpha_index in range(alpha_arr.shape[0]):
    for epoch_index in range(epoch_arr.shape[0]):
        obj1 = Neuron1D('training_data.txt', epoch_arr[epoch_index], 50, alpha_arr[alpha_index])
        obj1.train()
        if obj1.test() < min_TE:
            temp = min_TE
            delta_E = min_TE - obj1.test()
            best_alpha = obj1.alpha
            best_epoch = obj1.max_epoch
            if delta_E < 5:
                break

print(alpha_arr)
print(epoch_arr)



# obj2 = Neuron2D('training_data.txt', 10000, 0.05, 0.0001)
# obj2.train()
# obj2.test()
# obj3 = Neuron3D('training_data.txt', 10000, 0.05, 0.001)
# obj3.train()
# obj3.test()









