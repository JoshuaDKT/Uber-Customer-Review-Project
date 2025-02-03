import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv('../../data/uber_reviews_without_reviewid.csv')

# Retrieve basic dataset information
print("Shape of dataset:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print(df.sort(by='thumbsUpCount', ascending=False))

# Missing Values Analysis
missing_values = df.isnull().sum()
missing_percentages = (missing_values / len(df)) * 100
print("\nMissing values:\n", pd.DataFrame({
    "Missing values:": missing_values,
    "Percentage:": missing_percentages
}))

# Basic statistics
print("\nNumerical Statistics:\n", df.describe())
print("\nCategorical Statistics:\n", df.describe(include=['object']))

# 4. Unique Values in Each Column
for column in df.columns:
    n_unique = df[column].nunique()
    print(f"\n{column}: {n_unique} unique values")
    # For categorical columns with few unique values, show value counts
    if n_unique < 10 and df[column].dtype == 'object':
        print(df[column].value_counts())

# 5. Check for Duplicates
duplicate_count = df.duplicated().sum() # Get the sum of all rows that have duplicates in the dataset
print(f"\nNumber of duplicate rows: {duplicate_count}") # Print the number of duplicated rows

# 6. Basic Distribution Analysis for Numerical Columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns # Select columns with int and float values
for col in numerical_cols: # Iterate through each column in the numerical columns set
    print(f"\nSkewness of {col}: {df[col].skew()}") # Print the skewness of each column
    print(f"Kurtosis of {col}: {df[col].kurtosis()}") # Print the kurtosis of each column in the data