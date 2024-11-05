# Visualizing Missing Values
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
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


import matplotlib.pyplot as plt
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for feature in numeric_cols:
    print(feature)  # Replace with any operation you want to perform on each feature

for feature in numeric_cols:
    plt.figure(figsize=(8, 4))
    plt.hist(df[feature].dropna(), bins=30, edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--')
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

# Scaling Numerical Features
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Handling Categorical Features with Advanced Encoding (Target Encoding)
import category_encoders as ce
encoder = ce.TargetEncoder(cols=['vehicle_type', 'occupation'])
X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

# Correlation Matrix
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
