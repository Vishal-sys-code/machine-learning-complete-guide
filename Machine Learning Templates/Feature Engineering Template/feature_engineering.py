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
    # Step 1: Printing or Displaying the missing values of each column in %
    missing_values = df.isnull().sum()
    print("Missing Values are (in %): ")
    print(missing_values)

    # Step 2: Selecting the columns with less than 5% missing values
    cols = [var for var in df.columns if df[var].isnull().mean() < 0.05 and df[var].isnull().mean() > 0]
    print("Selected columns who have less than 5% missing values: ")
    print(cols)

    # Step 3: Applying Complete Case Analysis
    new_df = df[cols].dropna()
    print("Old Dataframe Shape: ", df.shape)
    print("New Dataframe Shape: ", new_df.shape)

    # Step 5: Visualise the distributions Before and After Complete Case Analysis
    for col in cols:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Original data (Red Color)
        df[col].hist(bins = 50, ax = ax, color = 'red', density = True, alpha = 0.5, label = f'Original {col}')
        # Data after CCA (Green Color)
        new_df[col].hist(bins = 50, ax = ax, color = 'green', density = True, alpha = 0.5, label = f'New {col}')
        ax.set_title(f'Histogram of {col}')
        ax.set.xlabel(col)
        ax.set_ylabel('Density')
        ax.legend()
        plt.show()
    
    # Step 6: Comparing Category Proportions Before and After Complete Case Analysis

    # Initialising an empty list to hold the results
    results = []
    # Loop through each column in the selected columns list
    for col in cols:
        # Ensure the column is categorical before processing
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Calculate the proportions of both the original and transformed data
            temp = pd.concat([
                df[col].value_counts()/len(df), # Original Data
                new_df[col].value_counts()/len(new_df) # Transformed Data
            ], axis = 1)
            
            # Add Columns Names
            temp.columns = ['Original', 'Transformed']
            # Add a Column Name to indicate which column we are comparing
            temp['column'] = col
            # Append the result to the results list
            results.append(temp)
        
    # Concatenate all results into one dataframe
    comparison_df = pd.concat(reults)
    # Display the final comparison Dataframe
    print(comparison_df)