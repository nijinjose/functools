import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

# **1. Prepare the Data**

# Assuming 'df' is your DataFrame with one-hot encoded features and a 'target' column
# Separate features (X) and the target variable (y)
X = df.drop(columns=['target'])
y = df['target']

# **2. Split the Data into Training and Testing Sets**

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 20% of data goes into the test set
    random_state=42,        # Seed for reproducibility
    stratify=y              # Maintains class distribution in splits
)

# **3. Create LightGBM Dataset Objects**

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# **4. Set up the LightGBM Parameters**

params = {
    'objective': 'binary',       # Binary classification
    'metric': 'auc',             # Use AUC for evaluation
    'boosting_type': 'gbdt',     # Gradient Boosting Decision Trees
    'learning_rate': 0.05,       # Learning rate
    'num_leaves': 31,            # Max number of leaves in one tree
    'feature_fraction': 0.8,     # Fraction of features to consider per iteration
    'bagging_fraction': 0.8,     # Fraction of data to consider per iteration
    'bagging_freq': 5,           # Bagging frequency
    'is_unbalance': True,        # Handle unbalanced classes
    'seed': 42                   # Seed for reproducibility
}

# **5. Define the Early Stopping Callback**

callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]

# **6. Train the LightGBM Model**

print("Training the model...")
model = lgb.train(
    params,                       # Parameters defined above
    train_data,                   # Training data
    num_boost_round=1000,         # Maximum number of boosting iterations (trees)
    valid_sets=[test_data, train_data],  # Validation data first for monitoring
    callbacks=callbacks,          # Early stopping callback
    verbose_eval=50               # Print progress every 50 iterations
)

# **7. Make Predictions on the Test Set**

y_pred_prob = model.predict(X_test)

# **8. Evaluate the Model**

auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC: {auc:.4f}")

# **9. Optimize the Decision Threshold for F1-Score**

def optimize_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    optimal_idx = f1_scores.argmax()
    return thresholds[optimal_idx]

threshold = optimize_threshold(y_test, y_pred_prob)
y_pred_optimized = (y_pred_prob >= threshold).astype(int)
f1_optimized = f1_score(y_test, y_pred_optimized)
print(f"Optimized F1-Score: {f1_optimized:.4f}, Threshold: {threshold:.4f}")

# **10. Print a Summary of the Results**

print("\nSummary of Results:")
print(f"AUC: {auc:.4f}")
print(f"Optimized F1-Score: {f1_optimized:.4f}, Threshold: {threshold:.4f}")
