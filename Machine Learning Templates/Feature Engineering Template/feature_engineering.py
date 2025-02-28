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
from sklearn.impute import SimpleImputer, KNNImputer


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

            # 5. Impute with end of distribution in the df_end_of_distribution_imputed dataframe
            if df[col].min() > 0: 
                # Use the minimum value for positive skew
                end_value = df[col].min()
            else:
                # Use the maximum value for negative skew
                end_value = df[col].max()
            df_end_of_distribution_imputed[col].fillna(end_value, inplace = True)

    # Step 4: Plotting Before and After Imputation for Selected Columns
    for col in cols:
        if df[col].dtype != 'object':  # Ensure it's a numerical column
            fig, axes = plt.subplots(1, 6, figsize=(20, 5))
            fig.suptitle(f'Before and After Imputation for Column: {col}', fontsize=16)
            
            # Plot before imputation
            sns.histplot(df[col], kde=True, ax=axes[0], color='blue')
            axes[0].set_title(f'Before Imputation')

            # Plot mean imputation
            sns.histplot(df_mean_imputed[col], kde=True, ax=axes[1], color='green')
            axes[1].set_title(f'Mean Imputation')

            # Plot median imputation
            sns.histplot(df_median_imputed[col], kde=True, ax=axes[2], color='orange')
            axes[2].set_title(f'Median Imputation')

            # Plot arbitrary value imputation
            sns.histplot(df_arbitrary_imputed[col], kde=True, ax=axes[3], color='red')
            axes[3].set_title(f'Arbitrary Value Imputation')

            # Plot random value imputation
            sns.histplot(df_random_imputed[col], kde=True, ax=axes[4], color='purple')
            axes[4].set_title(f'Random Value Imputation')

            # Plot end of distribution imputation
            sns.histplot(df_end_of_dist_imputed[col], kde=True, ax=axes[5], color='brown')
            axes[5].set_title(f'End of Distribution Imputation')

            # Show the plot
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # Adjust title to fit
            plt.show()

    # Final DataFrames after imputation
    print("\nData after mean imputation:")
    print(df_mean_imputed.head())
    
    print("\nData after median imputation:")
    print(df_median_imputed.head())

    print("\nData after arbitrary value imputation:")
    print(df_arbitrary_imputed.head())

    print("\nData after random value imputation:")
    print(df_random_imputed.head())

    print("\nData after end of distribution imputation:")
    print(df_end_of_dist_imputed.head())
    
    # Returning all dataframes after different imputations
    return df_mean_imputed, df_median_imputed, df_arbitrary_imputed, df_random_imputed, df_end_of_dist_imputed


def bivariate_missing_value_imputation_numerical(df, n_neighbors=5):
    """
    Perform bivariate missing value imputation using KNN and MICE.
    
    Args:
    df: pandas DataFrame with missing values
    n_neighbors: number of neighbors to use for KNN imputation
    
    Returns:
    df_knn_imputed: DataFrame after KNN imputation
    df_mice_imputed: DataFrame after MICE imputation
    """
    # Step 1: Checking missing values
    print("Missing values (in %):")
    missing_values = (df.isnull().sum() / len(df)) * 100
    print(missing_values)

    # Step 2: KNN Imputation
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_knn_imputed = df.copy()
    df_knn_imputed[:] = knn_imputer.fit_transform(df)  # Perform imputation
    print("\nData after KNN Imputation:")
    print(df_knn_imputed.head())

    # Step 3: MICE Imputation (using IterativeImputer from fancyimpute)
    mice_imputer = IterativeImputer(max_iter=10, random_state=0)
    df_mice_imputed = df.copy()
    df_mice_imputed[:] = mice_imputer.fit_transform(df)  # Perform imputation
    print("\nData after MICE Imputation:")
    print(df_mice_imputed.head())

    # Step 4: Visualize Before and After Imputation for Selected Columns
    cols_with_missing = [col for col in df.columns if df[col].isnull().sum() > 0]

    for col in cols_with_missing:
        if df[col].dtype != 'object':  # Ensure it's a numerical column
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'Before and After Imputation for Column: {col}', fontsize=16)
            
            # Plot before imputation
            axes[0].hist(df[col].dropna(), bins=20, color='blue', edgecolor='black')
            axes[0].set_title(f'Before Imputation')
            
            # Plot after KNN imputation
            axes[1].hist(df_knn_imputed[col], bins=20, color='green', edgecolor='black')
            axes[1].set_title(f'KNN Imputation')
            
            # Plot after MICE imputation
            axes[2].hist(df_mice_imputed[col], bins=20, color='orange', edgecolor='black')
            axes[2].set_title(f'MICE Imputation')
            
            # Show the plot
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # Adjust title to fit
            plt.show()

    # Returning both dataframes after imputation
    return df_knn_imputed, df_mice_imputed