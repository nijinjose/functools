# Cell 1: Import Libraries and Set Display Options
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Configure pandas display options for large datasets
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 60)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Set plot style and size
plt.style.use('classic')
plt.rcParams['figure.figsize'] = (12, 8)

# Cell 2: Load Data and Create Sample
# Load your data
# df = pd.read_csv('your_file.csv')

# Create a sample for visualization
sample_size = 10000
df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

# Cell 3: Initial Data Overview
print(f"Dataset Shape: {df.shape}")
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]:,}")

# Identify column types
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns

print("\nColumn Type Distribution:")
print(f"Numerical columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

print("\nMemory Usage:")
print(f"{df.memory_usage().sum() / 1024**2:.2f} MB")

# Cell 4: Column Analysis
# Create detailed column analysis
column_analysis = pd.DataFrame({
    'dtype': df.dtypes,
    'non_null_count': df.count(),
    'null_count': df.isnull().sum(),
    'null_percentage': (df.isnull().sum() / len(df) * 100).round(2),
    'unique_count': df.nunique(),
    'memory_MB': df.memory_usage(deep=True) / 1024**2
})

column_analysis['unique_percentage'] = (column_analysis['unique_count'] / len(df) * 100).round(2)
column_analysis['column_type'] = column_analysis['dtype'].apply(
    lambda x: 'Numeric' if pd.api.types.is_numeric_dtype(x) 
    else 'Categorical' if pd.api.types.is_object_dtype(x) or pd.api.types.is_categorical_dtype(x)
    else 'Other'
)

print("\nColumn Analysis (Top 20 by memory usage):")
display(column_analysis.sort_values('memory_MB', ascending=False).head(20))

# Cell 5: Missing Values Analysis
# Calculate missing values statistics
missing_stats = column_analysis[['null_count', 'null_percentage', 'column_type']].copy()
missing_stats = missing_stats[missing_stats['null_count'] > 0]

print("\nMissing Values Summary by Column Type:")
print(missing_stats.groupby('column_type')['null_count'].agg(['count', 'mean', 'max']))

print("\nTop 20 Columns with Missing Values:")
display(missing_stats.sort_values('null_count', ascending=False).head(20))

# Cell 6: Numeric Data Analysis
# Calculate summary statistics for numeric columns
print("\nNumeric Columns Summary Statistics:")
numeric_summary = df[numeric_cols].describe().round(3)
display(numeric_summary)

# Plot distributions for first 12 numeric columns (or less if fewer exist)
cols_to_plot = numeric_cols[:min(12, len(numeric_cols))]

for i in range(0, len(cols_to_plot), 3):
    plt.figure(figsize=(15, 5))
    for j, col in enumerate(cols_to_plot[i:i+3]):
        plt.subplot(1, 3, j+1)
        plt.hist(df_sample[col].dropna(), bins=30, edgecolor='black')
        plt.title(f'{col}\nDistribution')
        plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# Cell 7: Categorical Data Analysis
print("\nCategorical Columns Summary:")
for col in categorical_cols[:20]:  # First 20 categorical columns
    print(f"\n{col}:")
    n_unique = df[col].nunique()
    print(f"Unique values: {n_unique}")
    
    if n_unique <= 10:
        print("Complete value counts:")
        display(df[col].value_counts())
    else:
        print(f"Top 10 values (out of {n_unique}):")
        display(df[col].value_counts().head(10))
        
    # Calculate percentage of missing values
    missing_pct = (df[col].isnull().sum() / len(df) * 100).round(2)
    print(f"Missing values: {missing_pct}%")

# Cell 8: Correlation Analysis for Numeric Columns
# Calculate correlations using sample data
print("\nCalculating correlations for numeric columns...")
correlation_matrix = df_sample[numeric_cols].corr()

# Plot correlation matrix
plt.figure(figsize=(12, 8))
plt.imshow(correlation_matrix, cmap='RdBu', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix Heatmap (Sample Data)')
plt.tight_layout()
plt.show()

# Find and display top correlations
correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        correlations.append({
            'col1': correlation_matrix.index[i],
            'col2': correlation_matrix.columns[j],
            'correlation': abs(correlation_matrix.iloc[i, j])
        })

top_correlations = pd.DataFrame(correlations)
top_correlations = top_correlations.sort_values('correlation', ascending=False)

print("\nTop 15 Strongest Correlations:")
display(top_correlations.head(15))

# Cell 9: Categorical Relationships
# Analyze relationships between categorical variables with few unique values
analyzable_cats = [col for col in categorical_cols 
                  if df_sample[col].nunique() <= 10][:5]

if len(analyzable_cats) >= 2:
    for i in range(len(analyzable_cats)):
        for j in range(i+1, len(analyzable_cats)):
            col1, col2 = analyzable_cats[i], analyzable_cats[j]
            print(f"\nContingency Table: {col1} vs {col2}")
            cont_table = pd.crosstab(
                df_sample[col1],
                df_sample[col2],
                normalize='index'
            ).round(2)
            display(cont_table)

# Cell 10: Data Quality Summary
print("\nData Quality Summary:")

# Missing values summary
high_missing = column_analysis[column_analysis['null_percentage'] > 20]
print(f"\nColumns with >20% missing values: {len(high_missing)}")
if not high_missing.empty:
    print("\nTop 10 columns with high missing values:")
    display(high_missing[['null_percentage']].head(10))

# Cardinality check for categorical columns
print("\nCategorical Columns with High Cardinality:")
for col in categorical_cols:
    unique_ratio = df[col].nunique() / len(df)
    if unique_ratio > 0.5:
        print(f"{col}: {df[col].nunique():,} unique values ({unique_ratio:.1%} of rows)")

# Memory usage by type
print("\nMemory Usage by Column Type:")
memory_by_type = column_analysis.groupby('column_type')['memory_MB'].agg(['sum', 'mean'])
display(memory_by_type)
