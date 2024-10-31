
# Stage 1: Data Loading

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset


# Stage 2: Data Overview

# Display the first few rows of the dataframe
df.head()

# Basic information about the dataset
df.info()

# Summary statistics
df.describe()

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# Stage 3: Data Cleaning

# Handling missing values
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Plot missing values
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values.values)
plt.xticks(rotation=90)
plt.title('Missing Values per Column')
plt.ylabel('Number of Missing Values')
plt.show()

# Stage 4: Univariate Analysis

# Plot distributions of numerical features
for col in numerical_cols:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Plot categorical features
for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=df[col])
    plt.xticks(rotation=90)
    plt.title(f'Count of {col}')
    plt.show()

# Stage 5: Bivariate Analysis

# Analyzing relationships between features and the target variable
# Assuming 'cancellation_flag' is the target variable
target = 'cancellation_flag'

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Boxplots of numerical features against the target
for col in numerical_cols:
    if col != target:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df[target], y=df[col])
        plt.title(f'{col} vs {target}')
        plt.show()

# Stage 6: Categorical Features vs Target

# Analyzing categorical features with the target variable
for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=df[col], hue=df[target])
    plt.xticks(rotation=90)
    plt.title(f'{col} vs {target}')
    plt.show()

# Stage 7: Feature Relationships

# Pairplot for selected features
selected_features = numerical_cols[:4].tolist() + [target]
sns.pairplot(df[selected_features], hue=target)
plt.show()

# Stage 8: Statistical Testing

# Chi-square tests for categorical variables
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

print("Chi-square Test Results:")
for col in categorical_cols:
    contingency_table = pd.crosstab(df[col], df[target])
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    print(f"{col} vs {target}: chi2 = {chi2:.2f}, p-value = {p:.4f}")

# T-tests or ANOVA for numerical variables
print("\nT-test/ANOVA Results:")
for col in numerical_cols:
    if col != target:
        groups = [df[col][df[target] == value] for value in df[target].unique()]
        if len(groups) == 2:
            t_stat, p_value = ttest_ind(*groups)
            print(f"{col} vs {target}: t-stat = {t_stat:.2f}, p-value = {p_value:.4f}")
        else:
            f_stat, p_value = f_oneway(*groups)
            print(f"{col} vs {target}: F-stat = {f_stat:.2f}, p-value = {p_value:.4f}")

