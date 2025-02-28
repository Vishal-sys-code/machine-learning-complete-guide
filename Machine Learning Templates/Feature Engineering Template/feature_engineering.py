##################################
#  FEATURE ENGINEERING TEMPLATE  #
##################################

# Importing Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


##################################
#     FEATURE TRANSFORMATION     #
##################################

# Step 1: Missing Value Imputation

'''
-------- Contents --------
1.1 Complete Case Analysis
1.2 Missing Value -> Categorical Variables
1.3 Missing Value ->  Numerical Variables
1.4 Random Sample Imputer, Missing Indicator, Auto Select Imputer, KNN Imputer
1.5 Iterative Imputer
'''

# 1.1 Complete Case Analysis
"""
In complete case analysis, we find all the missing values in the form of the percentage.
If we get the missing values then we can drop the rows or columns.
We always drop the rows if the missing values are less than 5%.
"""

def complete_case_analysis():
    # Printing or Displaying the missing values of each column in %
    missing_values = df.isnull().sum()
    print("Missing Values are (in %): ")
    print(missing_values)

    # Selecting the columns with less than 5% missing values
    cols = [var for var in df.columns if df[var].isnull().mean() < 0.05 and df[var].isnull().mean() > 0]
    print("Selected columns who have less than 5% missing values: ")
    print(cols)

    # Applying Complete Case Analysis
    new_df = df[cols].dropna()
    print("Old Dataframe Shape: ", df.shape)
    print("New Dataframe Shape: ", new_df.shape)

    # Visualise the distributions Before and After Complete Case Analysis