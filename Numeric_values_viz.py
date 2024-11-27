import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Identify numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Get descriptive statistics
print(df[numerical_cols].describe())

# Visualize distributions
for col in numerical_cols:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
