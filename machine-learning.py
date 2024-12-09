"""
------------------ IMPLEMENTED ALGORITHMS ------------------
1. LINEAR REGRESSION [1 DIMENSIONAL]
2. MULTI LINEAR REGRESSION [HIGHER DIMENSIONAL]
3. GRADIENT DESCENT ALGORITHM
4. BATCH GRADIENT DESCENT ALGORITHM
5. STOCHASTIC GRADIENT DESCENT ALGORITHM
"""

# Important import necessary before starting a notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# ---------------------------- LINEAR REGRESSION [1 DIMENSIONAL] ----------------------------
# Custom Linear Regression [1 Dimensional]
class customLinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None
    def fit(self, X_train, y_train):
        numerator = 0
        denomenator = 0
        numerator = numerator + ((X_train[i]-X_train.mean())(y_train[i]-y_train.mean())) # ((Xi - X') * (Yi - Y'))
        denomenator = denomenator + ((X_train[i]-X_train.mean) * (X_train[i]-X_train.mean)) # ((Xi - X') * (Xi - X'))
        self.slope = (numerator/denomenator)
        self.intercept = (y_train.mean() - self.slope * (X_train.mean()))
    def predict(self, X_test):
        return (self.slope * X_test + self.intercept) # y = mx + c
    
# ---------------------------- MULTI LINEAR REGRESSION [HIGHER DIMENSIONAL] ----------------------------
# Custom Linear Regression [Higher Dimensional] 
# Applicable to all dimensions [1D, 2D, 3D, 4D, 5D, ..............., ND]
"""
Basic Knowledge:
- axis = 1: Apply functions in column wise 
- axis = 0: Apply functions in row wise 

We will do:
1. Add 1's to the starting column of the X_train
2. Use np.insert, np.linalg.inv and np.dot

Syntax:
np.insert(which_array, array_index, value_you_wanna_enter, axis = None or 1 or 0)
np.linalg.inv(array_name)
np.dot(array_name_1, array_name_2)

Mathematical Formula:
- beta = (X^T.X)^-1.X^T.Y 
- y = beta_0 + beta_1.X_1 + beta_2.X_2 + ........ + beta_n.X_n [This formula will go to the preidction]
- beta_1, beta_2, beta_3, ......... beta_n = coefficients
- beta_0 = intercept

Code Syntax: 
- intercept = beta[0]
- coefficients = beta[1:]
- prediction = np.dot(X_train, coefficients) + intercept

"""
class customMultiLinearRegression:
    def __init__(self):
        self.coefficient = None
        self.intercept = None
    def fit(self, X_train, y_train):
        X_train = np.insert(X_train, 0, 1, axis = 1)
        beta = np.linalg.inv(np.dot(X_train.T.X)).dot(X_train.T).dot(Y)
        intercept = beta[0]
        coefficient = beta[1:]
    def predict(X_test):
        return (np.dot(X_train, self.coefficient) + self.intercept)
    

# ---------------------------- GRADIENT DESCENT ALGORITHM ----------------------------
# Custom Gradient Descent Algorithm
"""
Gradient Descent Algorithm: 

We know the loss function is: 
loss_function(slope,intercept) = summation_1toN((y_i - y'_i)^2)

As we know in the gradient descent we need to derivate this above equation with respect to the slope and intercept one by one and that is called the loss slope. 
After this we will update both the values of intercept and the slope with a factor of learning rate and this loss slope. 

loss_slope_with_respect_to_intercept = (-2 * Summation_1_to_N(y_i - (slope * x_i) - intercept))
intercept_new = intercept_old - (learning_rate * loss_slope_with_respect_to_intercept)

loss_slope_with_respect_to_slope = (-2 * Summation_1_to_N(y_i - (slope * x_i) - intercept) * x_i)
slope_new = slope_old - (learning_rate * loss_slope_with_respect_to_slope)

At last, Prediction => slope * X + intercept
"""

class GradientDescentRegressor:
    def __init__ (self, learning_rate, epochs):
        self.slope = 100
        self.intercept = -120
        self.learning_rate = learning_rate
        self.epochs = epochs
    def fit(self, X, y):
        for i in range(epochs):
            loss_slope_with_respect_to_intercept = -2 * np.sum(y - (self.slope * X.ravel()) - self.intercept)
            loss_slope_with_respect_to_slope = -2 * np.sum((y - (self.slope * X.ravel()) - self.intercept) * X.ravel())
            self.intercept = self.intercept - (self.learning_rate * loss_slope_with_respect_to_intercept)
            self.slope = self.slope - (self.learning_rate * loss_slope_with_respect_to_slope)
        print(f"Loss in the slope w.r.t. intercept : {loss_slope_with_respect_to_intercept}")
        print(f"Loss in the slope w.r.t. slope : {loss_slope_with_respect_to_slope}")
        print(f"Intercepts;", self.intercept)
        print(f"Slopes;", self.slope)
    def predict(Self, X):
        return (self.slope * X.ravel() + self.intercept)
    

# ---------------------------- BATCH GRADIENT DESCENT ALGORITHM ----------------------------    
"""
Batch Gradient Descent Algorithm:

Common Terms:
- coefficients = beta_1, beta_2, beta_3, beta_4, ........., beta_n = Slope
- intercepts = beta_0 = Intercept
- X_train.shape[0] = Rows and X_train.shape[1] = Columns
- X_train.shape[0] = N and X_train.shape[1] = all the coefficients
- y = y_train, y_hat = np.dot(X_train, coefficient) + intercept
- loss_slope_with_respect_to_intercept(dL/dbeta_0) = (-2/n) summation_i_1toN(y_i - y_hat_i)  = -2 * (np.mean(y_train - y_hat))
- loss_slope_with_respect_to_coefficientN(dL/dbeta_N) = (-2/n) (summation_i_1toN(y_i - y_hat_i) * X_i1)  = -2 * (np.dot((y_train, y_hat), X_train)/X_train.shape[0])

This algorithm suggests that:
Step 1: Having a Random value of the coefficient = 1 and the intercepts = 0. [beta_0 = 0 and beta_1, beta_2, ..., beta_n = 1]
Step 2: Fixing the values of the epochs and the learning rates.
Step 3: Finding the loss slope with repect to the intercepts and the coefficient respectively.
Step 4: Updating the values of the intercepts and the coefficients with the new values.
In Prediction => follow: Y = mX + C => y_pred = np.dot(X_train, coefficients) + intercepts
"""
# Custom Batch Gradient Descent Algorithm
class BatchGradientDescentRegressor:
    def __init__(self, learning_rate, epochs):
        self.intercept = None
        self.coefficient = None
        self.learning_rate = learning_rate
        self.epochs = epochs
    def fit(self, X_train, y_train):
        # Step 1: Having a Random value of the coefficient = 1 and the intercepts = 0. [beta_0 = 0 and beta_1, beta_2, ..., beta_n = 1]
        self.intercept = 0
        self.coefficient = np.ones(X_train.shape[1])
        # Step 2: Fixing the values of the epochs and the learning rates.
        for i in range(self.epochs):
            # Step 3: Finding the loss slope with respect to the intercepts and the coefficient respectively.
            # Step 3.1: Calculating Y_hat
            y_hat = np.dot(X_train, coefficient) + intercept
            # Step 3.2: Updating the values of the intercept [beta_0]
            loss_slope_with_respect_to_intercept = -2 * (np.mean(y_train - y_hat))
            self.intercept = self.intercept - learning_rate * loss_slope_with_respect_to_intercept
            # Step 3.2: Updating the values of the Coefficients [beta_1, beta_2, beta_3, beta_4, ........., beta_n]
            loss_slope_with_respect_to_coefficient = -2 * (np.dot((y_train - y_hat), X_train)/ X_train.shape[0])
            self.coefficient = self.coefficient - learning_rate * loss_slope_with_respect_to_coefficient
        print("Intercept: ", intercept)
        print("Coefficient: ", coefficient)
    def predict():
        return (np.dot(X_train, self.coefficient) + self.intercept)

# ---------------------------- STOCHASTIC GRADIENT DESCENT ALGORITHM (The most used Algorithm) ----------------------------    
"""
Coded Algorithm:
- coefficients = beta_1, beta_2, beta_3, beta_4, ........., beta_n = Slope
- intercepts = beta_0 = Intercept
- X_train.shape[0] = Rows and X_train.shape[1] = Columns
- X_train.shape[0] = N and X_train.shape[1] = all the coefficients
- coefficients = np.ones(X_train.shape[1]) | intercept = 0
- Two nested loops => one for epochs and one for the rows or N 
  for i in range(epochs):
      for j in range(X_train.shape[0]):
          # CODE OF SGD
- Random Index Selection:
    idx = np.random.randint(0, X_train.shape[0]) 
- In general => (y = y_train), (y_hat = np.dot(X_train, coefficient) + intercept)
  In SGD -> X_train = X_train[idx] and now replace it.
  y_hat_sgd = np.dot(X_train[idx], coefficient) + intercept
- Loss Calculation:
    loss_calculation = y_train[idx] - y_hat_sgd
- Gradient Calculation:
    loss_slope_with_respect_to_intercept = -2 * (loss_calculation)
    loss_slope_with_respect_to_coefficient = -2 * np.dot(loss_calculation, X_train[idx])
- Updation of the Intercept and the Coefficients:
    intercept_new = intercept_old - learning_rate * loss_slope_with_respect_to_intercept
    coefficient_new = coefficient_old - learning_rate * loss_slope_with_respect_to_coefficient 
- Prediction => (Y = mX + C) => y_pred = np.dot(coefficient, X_test) + intercept
    
STOCHASTIC GRADIENT DESCENT ALGORITHM:
Step 1: Having a Random value of the coefficient = 1 and the intercepts = 0. [beta_0 = 0 and beta_1, beta_2, ..., beta_n = 1]
Step 2: Random Index Selection -> Selecting one random data point from the training set. (X_train.shape[0])
Step 3: Prediction Calculation -> Calculate the value of y_hat. => (Follow the formula of y_hat_sgd)
Step 4: Loss Calculation -> loss_calculation
Step 5: Gradient Calculation -> Calculate the loss slope with respect to both the intercept and coefficients
Step 6: Updation of the Intercept and the Coefficients:
        6.1 : intercept_new =  intercept_old - (learning_rate * loss_slope_with_respect_to_intercept)
        6.2 : coefficient_new = coefficient_old - (learning_rate * loss_slope_with_respect_to_coefficient)
"""

class StochasticGradientDescentRegressor:
    def __init__(self, learning_rate, epoch):
        self.intercept = None
        self.coefficient = None
        self.learning_rate = learning_rate
        self.epoch = epoch
    def fit(self, X_train, y_train):
        self.intercept = 0
        self.coefficient = np.ones(X_train.shape[1])
        for i in range(self.epoch):
            for j in range(X_train.shape[0]):
                idx = np.random.randint(0, X_train.shape[0])
                y_hat_sgd = np.dot(X_train[idx], self.coefficient) + self.intercept
                loss_calculation = y_train[idx] - y_hat_sgd
                loss_slope_with_respect_to_intercept = -2 * (loss_calculation)
                loss_slope_with_respect_to_coefficient = -2 * np.dot(loss_calculation, X_train[idx])
                self.intercept = self.intercept - learning_rate * loss_slope_with_respect_to_intercept
                self.coefficient = self.coefficient - learning_rate * loss_slope_with_respect_to_coefficient
        print("Coefficients: ", self.coefficient)
        print("Intercept: ", self.intercept)
                
    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coefficient) + self.intercept
        return f"Prediction: {y_pred}"
