##################################
#  FEATURE ENGINEERING TEMPLATE  #
# Author : Vishal Pandey
# Twitter: its_vayishu
# GitHub : Vishal-Sys-code
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

def complete_case_analysis(df):
    # Step 1: Printing or Displaying the missing values of each column in %
    missing_values = (df.isnull().sum()/len(df)) * 100
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
        ax.set_xlabel(col)
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
    comparison_df = pd.concat(results)
    # Display the final comparison Dataframe
    print(comparison_df)

# 1.2 Missing Value -> Numerical Variables
"""
There are many ways to handle the missing values in the numerical variables:
- Univariate: Mean, Median, Arbitrary Value, Random Values, End of Distribution 
- Bivariate: KNN Imputer, Iterative Imputer or MICE

When to use Mean/Median (two criterias):
- Missing completely at Random (MCAR)
- Missing values <= 5%

Arbitrary value: Impute the missing values with a fixed arbitrary value (0 or -999)
Random Values: Impute the missing values with random values from the distribution of the variable
End of Distribution: Impute the missing values with the minimum or maximum value in the column
"""
def univariate_missing_value_imputation_numerical(df, arbitrary_value = -999):
    # Step 1: Printing or Displaying the missing values of each column in %
    missing_values = (df.isnull().sum()/len(df)) * 100
    print("Missing Values are (in %): ")
    print(missing_values)

    # Step 2: Selecting the columns with less than 5% missing values
    cols = [var for var in df.columns if df[var].isnull().mean() < 0.05 and df[var].isnull().mean() > 0]
    print("Selected columns who have less than 5% missing values: ")
    print(cols)

    # Create the copies of the original dataframe for each imputation method
    df_mean_imputed = df.copy() # For mean imputation
    df_median_imputed = df.copy() # For median imputation
    df_arbitrary_imputed = df.copy() # For arbitrary value imputation
    df_random_imputed = df.copy() # For random value imputation
    df_end_of_distribution_imputed = df.copy() # For end of distribution imputation

    # Step 3: Impute the missing values with different techniques for columns that meet criteria
    for col in cols:
        if df[col].dtype != 'object': # Ensure that it's a numerical column
            # 1. Impute with mean in the df_mean_imputed dataframe
            df_mean_imputed[col].fillna(df_mean_imputed[col].mean(), inplace = True)

            # 2. Impute with median in the df_median_imputed dataframe
            df_median_imputed[col].fillna(df_median_imputed[col].median(), inplace = True)

            # 3. Impute with arbitrary value in the df_arbitrary_imputed dataframe
            df_arbitrary_imputed[col].fillna(arbitrary_value, inplace = True)

            # 4. Impute with random values in the df_random_imputed dataframe
            non_null_values = df[col].dropna()
            random_values = np.random.choice(non_null_values, size = df[col].isnull().sum())
            df_random_imputed.loc[df_random_imputed[col].isnull()] = random_values
            