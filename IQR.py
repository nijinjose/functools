import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Check for missing values in each column
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_percentage = (df.isnull().mean() * 100).sort_values(ascending=False)

# Combine counts and percentages
missing_data = pd.concat([missing_values, missing_percentage], axis=1, keys=['Total Missing', 'Percentage'])

# Display columns with missing values
print("Missing Values in Each Column:")
print(missing_data[missing_data['Total Missing'] > 0])

# Visualize missing values
import missingno as msno

# Bar chart of missing values
msno.bar(df)
plt.show()

# Heatmap of missing values
msno.heatmap(df)
plt.show()










# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Analyze categorical variables
for col in categorical_cols:
    num_unique = df[col].nunique()
    top_categories = df[col].value_counts(normalize=True).head()
    skewness = df[col].value_counts(normalize=True).values[0]
    
    print(f"\nAnalysis of Categorical Variable: {col}")
    print(f"Number of Unique Categories: {num_unique}")
    print("Top Categories (with percentages):")
    print(top_categories)
    
    # Check for high cardinality
    if num_unique > 20:
        print(f"Warning: {col} has high cardinality.")
    
    # Check for skewness
    if skewness > 0.7:
        print(f"Note: {col} is highly skewed towards one category.")
    
    # Plot the distribution if cardinality is manageable
    if num_unique <= 20:
        plt.figure(figsize=(10, 4))
        sns.countplot(x=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=90)
        plt.show()





# Identify numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Initialize a DataFrame to store analysis results
analysis_results = pd.DataFrame(columns=['Feature', 'Mean', 'Median', 'Std', 'Skewness', 'Kurtosis', 'Outliers'])

for col in numerical_cols:
    mean = df[col].mean()
    median = df[col].median()
    std = df[col].std()
    skewness = df[col].skew()
    kurtosis = df[col].kurtosis()
    
    # Detect outliers using IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    num_outliers = outliers.shape[0]
    
    # Append results
    analysis_results = analysis_results.append({
        'Feature': col,
        'Mean': mean,
        'Median': median,
        'Std': std,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Outliers': num_outliers
    }, ignore_index=True)
    
    # Verbose output
    print(f"\nAnalysis of Numerical Feature: {col}")
    print(f"Mean: {mean:.2f}, Median: {median:.2f}, Std: {std:.2f}")
    print(f"Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}")
    print(f"Number of Outliers: {num_outliers}")
    
    # Interpret skewness
    if skewness > 1:
        print("Note: The distribution is highly positively skewed.")
    elif skewness < -1:
        print("Note: The distribution is highly negatively skewed.")
    else:
        print("Note: The distribution is approximately symmetric.")
    
    # Visualize distributions
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f'Distribution of {col}')
    
    # Annotate skewness
    plt.annotate(f'Skewness: {skewness:.2f}', xy=(0.75, 0.9), xycoords='axes fraction')
    
    # Highlight outlier thresholds
    plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
    plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
    plt.legend()
    
    plt.show()

# Display summary analysis
print("\nSummary of Numerical Features:")
print(analysis_results)




# Compute correlation matrix
corr_matrix = df[numerical_cols].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Identify highly correlated pairs (correlation coefficient > 0.8)
threshold = 0.8
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            col_pair = (corr_matrix.columns[i], corr_matrix.columns[j])
            corr_value = corr_matrix.iloc[i, j]
            high_corr_pairs.append((col_pair, corr_value))

print("\nHighly Correlated Pairs (Correlation Coefficient > 0.8):")
for pair, value in high_corr_pairs:
    print(f"{pair[0]} and {pair[1]}: {value:.2f}")






