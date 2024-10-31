# Cell 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set default plot style and size
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (10, 6)

# Set display options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Cell 2: Load Data
# Replace 'your_file.csv' with your actual data file
# df = pd.read_csv('your_file.csv')

# Cell 3: Basic Data Overview
# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nFirst few rows of the dataset:")
display(df.head())

print("\nDataset Info:")
print(df.info())

# Cell 4: Check Missing Values
# Display missing values count for each column
missing_values = df.isnull().sum()
missing_percentages = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage Missing': missing_percentages
})
print("Missing Values Analysis:")
display(missing_data[missing_data['Missing Values'] > 0])

# Cell 5: Numeric Data Summary
# Generate descriptive statistics for numeric columns
print("Numeric Columns Summary:")
display(df.describe())

# Cell 6: Categorical Data Summary
# Analyze categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    display(pd.DataFrame({
        'Count': df[col].value_counts(),
        'Percentage': df[col].value_counts(normalize=True) * 100
    }))

# Cell 7: Distribution Plots for Numeric Variables
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    plt.figure(figsize=(15, 5))
    
    # Create subplot for histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    
    # Create subplot for box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
    
    plt.tight_layout()
    plt.show()

# Cell 8: Correlation Analysis
# Create correlation matrix for numeric columns
plt.figure(figsize=(12, 8))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Cell 9: Categorical Variable Visualizations
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Cell 10: Statistical Tests
# For numeric columns: Shapiro-Wilk test for normality
print("Normality Tests (Shapiro-Wilk):")
for col in numeric_cols:
    statistic, p_value = stats.shapiro(df[col].dropna())
    print(f"\n{col}:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print("Normal distribution?" if p_value > 0.05 else "Not normal distribution")

# Cell 11: Outlier Detection
# Calculate Z-scores for numeric columns
print("\nOutlier Detection using Z-scores:")
for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    outliers = len(z_scores[z_scores > 3])
    print(f"\n{col}:")
    print(f"Number of outliers (|Z-score| > 3): {outliers}")
    print(f"Percentage of outliers: {(outliers/len(df[col].dropna()))*100:.2f}%")
