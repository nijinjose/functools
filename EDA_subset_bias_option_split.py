## 2. Data Extraction and Preparation

```python
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
import ace_tools


# Handle Missing Values
# Drop columns with more than 50% missing values and impute other columns.
missing_threshold = 0.5
data = data[data.columns[data.isnull().mean() < missing_threshold]]
data.fillna(data.median(), inplace=True)  # Replace with median values

# Train-Test Split (stratify by target variable for balanced split)
X = data.drop(columns=['churn'])
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

## 3. Exploratory Data Analysis (EDA) and Data Cleaning

```python
import matplotlib.pyplot as plt
import seaborn as sns

# General Information about the Dataset
print("Dataset Information:\n")
data.info()
print("\nSummary Statistics:\n")
print(data.describe())

# Check for Imbalance in the Target Variable
plt.figure(figsize=(8, 6))
sns.countplot(x='churn', data=data)
plt.title("Class Distribution")
plt.xlabel("churn Flag")
plt.ylabel("Count")
plt.show()

# Visualizing Missing Values in Batches
batch_size = 50
num_batches = len(data.columns) // batch_size + 1

for i in range(num_batches):
    start_col = i * batch_size
    end_col = min((i + 1) * batch_size, len(data.columns))
    plt.figure(figsize=(12, 6))
    sns.heatmap(data.iloc[:, start_col:end_col].isnull(), cbar=False, cmap='viridis')
    plt.title(f"Missing Values Heatmap (Columns {start_col} to {end_col})")
    plt.show()

# Analyze Numerical Features
# Due to the large number of features (800), we'll sample a subset for visualization
sampled_numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns[:10]
for col in sampled_numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=X_train, x=col, kde=True, hue=y_train)
    plt.title(f"Distribution of {col} by Cancellation")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Analyze Categorical Features
categorical_columns = X_train.select_dtypes(include=['object']).columns[:10]  # Sample a subset due to high dimensionality
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=X_train, x=col, hue=y_train)
    plt.title(f"Distribution of {col} by Cancellation")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# Outlier Detection and Removal for Numerical Features
from sklearn.preprocessing import StandardScaler
from scipy import stats

numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns

# Z-score method for outlier detection
z_scores = np.abs(stats.zscore(X_train[numerical_columns]))
X_train = X_train[(z_scores < 3).all(axis=1)]
y_train = y_train[X_train.index]

# Scaling Numerical Features refer distiribution
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Handling Categorical Features with Advanced Encoding (Target Encoding) --> optional
import category_encoders as ce
encoder = ce.TargetEncoder(cols=['vehicle_type', 'occupation'])
X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

# Correlation Matrix ---> otpional
# Due to high dimensionality, we'll visualize a subset of features for correlation analysis
plt.figure(figsize=(12, 8))
corr_matrix = X_train.iloc[:, :20].corr()
sns.heatmap(corr_matrix, annot=False, cmap="viridis")
plt.title("Correlation Matrix (Sampled Features)")
plt.show()

# Cleaned Data Summary
print("Cleaned Dataset Information:\n")
X_train.info()
print("\nCleaned Summary Statistics:\n")
print(X_train.describe())
```
