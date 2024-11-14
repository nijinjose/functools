# Cell 1: Initial Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats

# Set display options
pd.set_option('display.max_columns', 20)
plt.style.use('classic')


# Create a holdout set right at the start
# This is data we'll never look at until final model evaluation
df_analysis, df_holdout = train_test_split(
    df, 
    test_size=0.2,  # 20% holdout
    random_state=42,
    stratify=df['cancelled']  # Ensure balanced split
)

print("Dataset Splits:")
print(f"Analysis set size: {len(df_analysis):,} ({len(df_analysis)/len(df)*100:.1f}%)")
print(f"Holdout set size: {len(df_holdout):,} ({len(df_holdout)/len(df)*100:.1f}%)")

# Verify stratification worked
print("\nCancellation Rates in Splits:")
print(f"Original data: {df['cancelled'].mean()*100:.2f}%")
print(f"Analysis set: {df_analysis['cancelled'].mean()*100:.2f}%")
print(f"Holdout set: {df_holdout['cancelled'].mean()*100:.2f}%")

# Cell 3: EDA on Analysis Set
print("\nPerforming EDA on Analysis Set Only:")

# Basic statistics
print("\nBasic Statistics (Analysis Set):")
print(f"Total policies: {len(df_analysis):,}")
print(f"Cancellation rate: {df_analysis['cancelled'].mean()*100:.2f}%")

# Example of proper feature analysis
numeric_cols = df_analysis.select_dtypes(include=['int64', 'float64']).columns
numeric_cols = numeric_cols.drop(['cancelled']) if 'cancelled' in numeric_cols else numeric_cols

# Analyze one numeric feature as example
for col in numeric_cols[:1]:  # Just showing one for brevity
    print(f"\nAnalyzing {col}:")
    
    # Get statistics
    stats = df_analysis[col].describe()
    print("\nFeature Statistics (Analysis Set Only):")
    display(stats)
    
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.hist(df_analysis[col], bins=30)
    plt.title(f'Distribution of {col} (Analysis Set)')
    plt.show()

# Cell 4: Feature Engineering (on analysis set)
def create_features(data):
    """Create new features - will apply to both analysis and holdout sets"""
    data = data.copy()
    
    # Example feature engineering
    if 'policy_start_date' in data.columns:
        data['policy_month'] = pd.to_datetime(data['policy_start_date']).dt.month
    
    # Add more feature engineering as needed
    return data

# Apply feature engineering to analysis set only
df_analysis_featured = create_features(df_analysis)

# Cell 5: Model Development Split
# Now split analysis set into train/validation
X = df_analysis_featured.drop('cancelled', axis=1)
y = df_analysis_featured['cancelled']

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.25,  # 25% of analysis set = 20% of original data
    random_state=42,
    stratify=y
)

print("\nFinal Dataset Splits:")
print(f"Training set: {len(X_train):,} records")
print(f"Validation set: {len(X_val):,} records")
print(f"Holdout set: {len(df_holdout):,} records")

# Verify all splits maintain class distribution
print("\nCancellation Rates Across All Splits:")
print(f"Original data: {df['cancelled'].mean()*100:.2f}%")
print(f"Training set: {y_train.mean()*100:.2f}%")
print(f"Validation set: {y_val.mean()*100:.2f}%")
print(f"Holdout set: {df_holdout['cancelled'].mean()*100:.2f}%")

# Cell 6: Proper Preprocessing Workflow
from sklearn.preprocessing import StandardScaler

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation data using training set parameters
X_val_scaled = scaler.transform(X_val)

# Note: Holdout set remains untouched until final model evaluation

print("\nProper Preprocessing Workflow:")
print("1. Scaler fit on training data only")
print("2. Same scaler applied to validation data")
print("3. Holdout set remains untouched")

# Cell 7: Document Analysis Pipeline
print("\nComplete Analysis Pipeline:")
print("""
1. Initial Split:
   - Split full dataset into analysis (80%) and holdout (20%) sets
   - Stratify by target variable
   
2. Exploratory Data Analysis:
   - Perform EDA on analysis set only
   - Document all findings and decisions
   
3. Feature Engineering:
   - Develop feature engineering on analysis set
   - Document transformations for later use on holdout
   
4. Model Development Split:
   - Split analysis set into train (60% of total) and validation (20% of total)
   - Stratify by target variable
   
5. Preprocessing:
   - Fit preprocessors on training data only
   - Apply to validation data using training parameters
   
6. Model Development:
   - Train models using training data
   - Evaluate on validation data
   - Iterate as needed
   
7. Final Evaluation:
   - Only after model is finalized:
     - Apply feature engineering to holdout set
     - Apply preprocessing using training set parameters
     - Evaluate final model performance
""")
