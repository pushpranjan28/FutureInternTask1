import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv(r"C:\Users\LENOVO\Downloads\test.csv")

# Checking for missing values in each column
print("Missing values before cleaning:\n", df.isnull().sum())

# Removing rows with missing values
df_cleaned = df.dropna()

# Checking if missing values were removed
print("\nMissing values after cleaning:\n", df_cleaned.isnull().sum())

# Descriptive statistics for numerical columns (to spot outliers)
print("\nStatistics before outlier removal:\n", df_cleaned.describe())

# Defining a function to remove outliers using Z-score
def remove_outliers(df, col):
    mean = df[col].mean()
    std = df[col].std()
    z_score = (df[col] - mean) / std
    return df[(z_score > -3) & (z_score < 3)]

# Removing outliers from numerical columns
numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns

for col in numerical_cols:
    df_cleaned = remove_outliers(df_cleaned, col)

# Checking descriptive statistics after outlier removal
print("\nStatistics after outlier removal:\n", df_cleaned.describe())

# Saving the cleaned dataset
df_cleaned.to_csv('cleaned_titanic_dataset.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_titanic_dataset.csv'")
