"""
------------------ IMPLEMENTED DATA ANALYSIS ------------------
- IMPORTED IMPORTANT LIBRARIES
- PLAYING WITH DATA
- EXPLORATORY DATA ANALYSIS
- FEATURE ENGINEERING

------------------ IMPLEMENTED ALGORITHMS ------------------
1. LINEAR REGRESSION [1 DIMENSIONAL]
2. MULTI LINEAR REGRESSION [HIGHER DIMENSIONAL]
3. GRADIENT DESCENT ALGORITHM
4. BATCH GRADIENT DESCENT ALGORITHM
5. STOCHASTIC GRADIENT DESCENT ALGORITHM
6. MINI BATCH GRADIENT DESCENT ALGORITHM
7. POLYNOMIAL REGRESSION
8. RIDGE REGRESSION [2-DIMENSIONAL, N-DIMENSIONAL, GRADIENT DESCENT]
"""

# ---------------------------- IMPORTANT IMPORT NECESSARY LIBRARIES BEFORE STARTING A NOTEBOOK ----------------------------
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport # Powerful Automatic EDA
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from sklearn import datasets 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# ---------------------------- PLAYING WITH DATA [PANDAS] ----------------------------

# If you have local csv file
df = pd.read_csv('path_of_your_csv_files.csv')

# If you are fetching csv file from the URL
url = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
headers = {"USer-Agent": "Mozilla/5.0 (Macintosh; Intel Max OS X 10.14; rv:66.0) Gecko/20100101 FireFox/66.0"}
req = requests.get(url, headers = headers)
data = StringIO(req.text)
pd.read_csv(data)

"""
Pandas Parameter to play with data. When to use:
# 1. sep Parameter:
# Use this when the data is separated by a custom delimiter (other than commas).
# Example: sep="\t" for tab-separated values, or sep="|" for pipe-separated values.

# 2. index_col Parameter:
# Use this when you want one or more columns to be used as the index of the DataFrame.
# Example: index_col=0 to set the first column as the index.

# 3. header Parameter:
# Use this to specify which row to use as column names (header).
# Default is header=0 (first row). For example, header=1 if the header is in the second row.

# 4. usecols Parameter:
# Use this to read specific columns from the dataset.
# Example: usecols=["col1", "col2"] to only read the specified columns.

# 5. squeeze Parameter:
# Use this when you want to reduce the output from a DataFrame to a Series, 
# if only a single column is selected. Example: squeeze=True.

# 6. skiprows / nrows Parameters:
# - skiprows: Skip specific rows at the beginning. Example: skiprows=5 to skip the first 5 rows.
# - nrows: Limit the number of rows to load. Example: nrows=100 to load only the first 100 rows.

# 7. encoding Parameter:
# Use this when your dataset has a specific encoding (e.g., UTF-8, ISO-8859-1).
# It's especially useful for datasets with special characters or non-English text.

# 8. skip_bad_lines (or error_bad_lines in older versions):
# Use this when your data contains corrupt lines that you want to ignore while loading.
# Example: error_bad_lines=False to skip bad lines.

# 9. dtype Parameter:
# Use this to specify the data types for certain columns, especially if pandas' default type inference is incorrect.
# Example: dtype={"age": int, "salary": float}.

# 10. Handling Dates:
# Use parse_dates to automatically convert date columns to datetime objects.
# You can also define a custom date parser with date_parser if needed.
# Example: parse_dates=["date_column"] to parse a date column.

# 11. converters Parameter:
# Use this to apply custom functions to columns while reading the file.
# Example: converters={"column_name": custom_function} to transform data on the fly.

# 12. na_values Parameter:
# Use this to specify additional strings to be treated as missing (NaN) values.
# Example: na_values=["N/A", "NULL"] to treat these strings as NaN.

# 13. Loading Huge Datasets in Chunks:
# Use chunksize to load large datasets in smaller, manageable parts.
# Example: chunksize=1000 to load data in chunks of 1000 rows at a time.
# This helps reduce memory usage and handle large files efficiently.
"""

# ---------------------------- ASKING 7 QUESTIONS WITH DATA BEFORE PERFORMING EDA ----------------------------
"""
1. How big is the data?  => df.shape
2. How does the data look like? => df.head()
3. What is the datatype of the columns? => df.info()
4. Are there any missing values? => df.isnull().sum() and df.isnull().mean()
5. How does the data looks like mathematically? => df.describe()
6. Are there any duplicate value? => df.duplicated().sum()
7. How is the correlation between columns? => df.corr()
"""
# 1. How big is the data?
df.shape
# 2. How does the data look like?
df.head()
# 3. What is the datatype of the columns?
df.info()
# 4. Are there any missing values?
df.isnull().sum()
# 5. How does the data looks like mathematically?
df.describe()
# 6. Are there any duplicate value?
df.duplicate().sum()
# 7. How is the correlation between columns?
df.corr()

# ---------------------------- EXPLORATORY DATA ANALYSIS ----------------------------
"""
Why we need Exploratory Data Analysis(EDA)?
-> We need EDA to understand and visualise the important characteristics of the data.
-> It also detect anamolies, testing assumptions and generate hypothesis.

EDA is of three type: Univariate Analysis, Bivariate Analysis and Multi-variate Analysis.
- Univariate Analysis: When we do independent analysis on 1 columns
- Bivariate Analysis: When we do independent analysis on 2 columns
- Multi-variate Analysis: When we do independent analysis on more than 2 columns

There are two types of data: Categorical and Numerical
- Numerical: height, weight, age, ...etc
- Categorical: country, gender, nationality, ...etc

Actually in EDA we plot the data distributions to check if the data or the column 
have the normal distribution or not. If they dont have the normal distribution 
then we perform the Feature Engineering on it to transform that data. 

We plot the Probability Density Function (PDF).
- if PDF is normal distributed then no need to change or transform it
- else we need to apply the feature engineering to transform the data.

NOTE: We actually do mostly Bivariate and MultiVariate Analysis.

For Univariate Analysis we do the plots like:
1. For Categorical Data: Countplot and Piechart
2. For Numerical Data: Histogram, DistPlot, BoxPlot.

For Bivariate or Multivariate Analysis we do the plots like:
We plot the data in the form of the (Numerical - Numerical), (Numerical - Categorical), (Categorical - Categorical)
We can say as => (N - N), (N - C), (C - C)

1. ScatterPlot: (Numerical - Numerical)
2. BarPlot:     (Numerical - Categorical)
3. BoxPlot:     (Numerical - Categorical)
4. DistPlot:    (Numerical - Categorical)
5. HeatMap:     (Categorical - Categorical)
6. ClusterMap:  (Categorical - Categorical)
7. PairPlot:    (Numerical - Numerical)
8. LinePlot:    (Numerical - Numerical)

(Numerical - Numerical): ScatterPlot, LinePlot, PairPlot
(Numerical - Categorical): BarPlot, BoxPlot, DistPlot
(Categorical - Categorical): HeatMap, ClusterMap

IMPORTANT NOTES:
If you find any probability distribution function plotting as the normal distribution, then it doesn't need to be transformed by the feature engineering.
If you find any probability distribution function plotting as the skewness distribution, then it need to be transformation with the help of the feature engineering.

You will get some of the Normal distributed data to be contained as the: missing values. This need to be fixed in feature engineering no doubt in this.

Normal Distribution: A good curve in the middle of the plot
Skewness Distribution: A curve that is shifted towards more to the left or to the right. 


PANDAS CONTAIN A VERY GREAT HANDY METHOD THAT DO THE EDA AUTOMATICALLY IN DETAILED WAY. IT IS KNOWN AS THE PANDAS PROFILING.
"""

# ---------------------------- PANDAS PROFILING [Automatic EDA] ----------------------------
df = pd.read_csv("local_csv_path.csv")
profile = ProfileReport(df, title = "Pandas Profiling Report", explorative = True)
profile.to_file('automated_eda.html')


# ---------------------------- FEATURE ENGINEERING ----------------------------


# ---------------------------- LINEAR REGRESSION [1 DIMENSIONAL] ----------------------------
# Custom Linear Regression [1 Dimensional]
class customLinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None
    def fit(self, X_train, y_train):
        numerator = 0
        denomenator = 0
        for i in range(X_train.shape[0]):
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
    
STOCHASTIC GRADIENT DESCENT PSEUDOCODE:
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

# ---------------------------- MINI BATCH GRADIENT DESCENT ALGORITHM ----------------------------

"""
Coded Algorithm:
- coefficients = beta_1, beta_2, beta_3, beta_4, ........., beta_n = Slope
- intercepts = beta_0 = Intercept
- X_train.shape[0] = Rows and X_train.shape[1] = Columns
- X_train.shape[0] = N and X_train.shape[1] = all the coefficients
- coefficients = np.ones(X_train.shape[1]) | intercept = 0
- Two nested loops => one for epochs and one for the rows or N 
  for i in range(epochs):
      for j in range(int(X_train.shape[0]/batch_size)):
          # CODE OF MBGD
- Random Selection of the Subsets of indices from the training data [Random Sampling of Mini Batch]:
    idx = random.sample(range(X_train.shape[0]), batch_size)
- Calculation of y_hat_mbgd:
    In general => (y = y_train), (y_hat = np.dot(X_train, coefficient) + intercept)
    However, In MBGD: X_train = X_train[idx]
    y_hat_mbgd = np.dot(X_train[idx], coefficient) + intercept
- Calculation of loss:
    loss_calculation = (y_train[idx] - y_train)
- Gradients Calculation:
    loss_slope_with_respect_to_intercept = -2 * np.mean(loss_calculation)
    loss_slope_with_respect_to_coefficient = -2 * np.dot(loss_calculation, X_train[idx])
- Update the Coefficients and the Intercepts:
    intercept_new = intercept_old - (learning_rate * loss_slope_with_respect_to_intercept)
    coefficient_new = coefficient_old - (learning_rate * loss_slope_with_respect_to_coefficient)
- Prediction => (Y = mX + C) => y_pred = np.dot(coefficient, X_test) + intercept

MINI BATCH GRADIENT DESCENT PSEUDOCODE:
Note:- A new variable will be introduced: batch_size. [A user input]
Step 1: Having a Random value of the coefficient = 1 and the intercepts = 0. [beta_0 = 0 and beta_1, beta_2, ..., beta_n = 1]
Step 2: Random Selection for Subset of Indices -> Selecting the random set of data points from the training set. (X_train.shape[0]/batch_size)
Step 3: Prediction Calculation -> Calculate the value of y_hat. => (Follow the formula of y_hat_mbgd)
Step 4: Loss Calculation -> loss_calculation
Step 5: Gradient Calculation -> Calculate the loss slope with respect to both the intercept and coefficients
Step 6: Updation of the Intercept and the Coefficients:
        6.1 : intercept_new =  intercept_old - (learning_rate * loss_slope_with_respect_to_intercept)
        6.2 : coefficient_new = coefficient_old - (learning_rate * loss_slope_with_respect_to_coefficient)
"""
class MiniBatchGradientDescentRegressor:
    def __init__(self, learning_rate, epochs, bactch_size):
        self.intercept = None
        self.coefficient = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
    def fit(self, X_train, y_train):
        self.intercept = 0
        self.coefficient = np.ones(X_train.shape[1])
        for i in range(self.epochs):
            for j in range(int(X_train.shape[0]/self.batch_size)):
                idx = random.sample(range(X_train.shape[0]), self.batch_size)
                y_hat_mbgd = np.dot(X_train[idx], self.coefficient) + self.intercept
                loss_calculation = (y_train[idx] - y_hat_mbgd)
                loss_slope_with_respect_to_intercept = -2 * np.mean(loss_calculation)
                loss_slope_with_respect_to_coefficient = -2 * np.dot((loss_calculation), X_train[idx])
                self.intercept = self.intercept - (self.learning_rate * loss_slope_with_respect_to_intercept)
                self.coefficient = self.coefficient - (self.learning_rate * loss_slope_with_respect_to_coefficient)
        print("Coefficients: ", self.coefficient)
        print(" ")
        print("Intercept: ", self.intercept)
    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coefficient) + self.intercept
        return y_pred

# ---------------------------- POLYNOMIAL REGRESSION ----------------------------

def polynomial_regression(degree):
    X_new = np.linspace(-3, 3, 100).reshape(100, 1)
    X_new_poly = poly.transform(X_new)
    polybig_features = PolynomialFeatures(degree = degree, include_bias = False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
        ('poly_features', polybig_features),
        ('std_scaler', std_scaler),
        ('lin_reg', lin_reg)
    ])
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    plt.plot(X_new, y_newbig, 'r', label = "Degree" + str(degree), linewidth = 2)
    plt.plot(X_train, y_train, 'b.', linewidth = 3)
    plt.plot(X_test, y_test, 'g.', linewidth = 3)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis([-3, 3, 0, 10])
    plt.legend(loc = "upper left")
    plt.show()

# ---------------------------- RIDGE REGRESSION [2 Dimensional] ----------------------------
"""
Ridge Regression in 2 dimensional is similar to the Simple Linear Regression. However there is term added in the Loss Function of the Ridge Regression which is the penality term.
Penalty term = λ ||w||^2. Here, w is the coefficients. Previously we have denoted w as coefficients. It actually work like weights.
The λ is the hyperparameter which is the regularization parameter. It is the hyperparameter which is used to control the strength of the penalty term. If the λ is high, then the penalty term is high and the model is regularized heavily. If the λ is low, then the penalty term is low and the model is regularized lightly.

Formula:
penalty_term =  λ (We call it as alpha)
numerator = summation_1toN ((y_i - y_mean) * (x_i - x_mean))
denominator = summation_1toN ((x_i - x_mean) ^ 2)
m (slope) = [(numerator) / (denomenator + penalty_term)]
b (intercept) = y_mean - m * (x_mean)
"""

class CustomSimpleRidgeRegression:
    def __init__(self, alpha):
        self.alpha = alpha
        self.coefficient = None
        self.intercept = None
    def fit(self, X_train, X_test):
        numerator = 0 
        denominator = 0 
        for i in range(X_train.shape[0]):
            numerator = numerator + ((y_train[i] - y_train.mean()) * (X_train[i] - X_train.mean()))
            denominator = denominator + (X_train[i] - X_train.mean()) ** 2
        self.coefficient = numerator / (denominator + self.alpha)
        self.intercept = y_train.mean() - (self.coefficient * X_train.mean())
        print("Coefficient: ", self.coefficient)
        print("Intercept: ", self.intercept)
    def predict(self, X_test):
        y_pred = 0 
        for i in range(X_train.shape[0]):
            y_pred = y_pred + np.dot(X_test[i], self.coefficinet) + self.intercept[i]
        return y_pred

# ---------------------------- RIDGE REGRESSION [N Dimensional] ----------------------------
"""
Formula:
I = Identity Matrix
w = (X_transpose.X + λ.I)^-1 (X_transpose.Y)

First_Term = (X_transpose.X + λ.I)^-1 => np.linalg.inv(np.dot(X_train.T, X_train) + np.dot(self.alpha, identity_matrix)
Second_Term = (X_transpose.Y) => np.dot(X_train.T, y_train)

w = First_Term + Second_Term



Common Terms:
- This is same as the Higher Dimensional Linear Regression however here is an additional term of the penalty.
- Inserting the 1's to the first row of the X_train
  - X_train = np.insert(X_train, 0, 1, axis = 1)
- Identity matrix = np.identity(X_train.shape[1])
- penalty_term = self.alpha * identity_matrix
- Calculate W from the above numpy explanation
- Set the values of the coefficient and intercept:
    - coefficient = w[1:]
    - intercept = w[0]
- Prediction => Y = mX + C => (coefficient, X_train) + intercept
"""

class CustomRidgeRegressionMultiDimensional:
    def __init__(self, alpha = 0.1):
        self.alpha = alpha
        self.coefficient = None
        self.intercept = None
    def fit():
        X_train = np.insert(X_train, 0, 1, axis = 1)
        identity_matrix = np.identity(X_train.shape[1])
        w = np.linalg.inv(np.dot(X_train.T,X_train) + np.dot(self.alpha, identity_matrix)).dot(X_train.T).dot(y_train)
        self.intercept = w[0]
        self.coefficient = w[1:]
        print('Coefficient: ', self.coefficient)
        print('Intercept: ', self.intercept)
    def predict():
        return np.dot(X_test, self.coefficient) + self.intercept


# If you want the exact result as the sklearn create (without any precision error) then use a small change
class CustomRidgeRegressionMultiDimensional_Sklearn:
    def __init__(self, alpha = 0.1):
        self.alpha = alpha
        self.coefficient = None
        self.intercept = None
    def fit():
        X_train = np.insert(X_train, 0, 1, axis = 1)
        identity_matrix = np.identity(X_train.shape[1])
        identity_matrix[0][0] = 0 # Small Change
        w = np.linalg.inv(np.dot(X_train.T,X_train) + np.dot(self.alpha, identity_matrix)).dot(X_train.T).dot(y_train)
        self.intercept = w[0]
        self.coefficient = w[1:]
        print('Coefficient: ', self.coefficient)
        print('Intercept: ', self.intercept)
    def predict():
        return np.dot(X_test, self.coefficient) + self.intercept


# ---------------------------- GRADIENT DESCENT FOR RIDGE REGRESSION ----------------------------
"""
It is same as we did in the previous Gradient Descents:

Coded Algorithm:
- coefficients = w_1, w_2, w_3, w_4, ........., w_n = Slope
- intercepts = w_0 = Intercept
- X_train.shape[0] = Rows and X_train.shape[1] = Columns
- X_train.shape[0] = N and X_train.shape[1] = all the Coefficients
- intercept = 0, coefficient = np.ones(X_train.shape[1])
- w = np.insert(self.coefficient, 0, self.intercept)
- X_train = np.insert(X_train, 0, 1, axis = 1)
- One Loop in the range of the epochs
  - Formula of w = (X_transpose . X . w) - (X_transpose * Y) + (alpha * w)
  - derivative_of_w = (np.dot(X_train.T, X_train).dot(w)) - np.dot(X_train.T, y_train) + (alpha * w)
  - w_new = w_old - (learning_rate * derivative_of_w)
- intercept = w[0]
- coefficient = w[1:]
- Prediction => Y = mx + C => np.dot(X_train * self.coefficient) + self.intercept
"""

class  CustomGradientDescentRidgeRegression:
    def __init__(self, epochs, learning_rate, alpha):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.intercept = None
        self.coefficient = None
    def fit(self):
        self.intercept = 0
        self.coefficient = np.ones(X_train.shape[1])
        w = np.insert(self.coefficient, 0, self.intercept)
        X_train = np.insert(X_train, 0, 1, axis = 1)
        for i in range(self.epochs):
            derivative_of_w = (np.dot(X_train.T, X_train).dot(w) - np.dot(X_train.T, y_train) + (self.alpha * w))
            w = w - (self.learning_rate * derivative_of_w)
        self.intercept = w[0]
        self.coefficient = w[1:]
        print("Coefficient: ", coefficient)
        print("Intercept: ", intercept)
    def predict(self):
        return np.dot(X_test, self.coefficient) + self.intercept
