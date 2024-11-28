import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve


# Split into features and target
X = df.drop(columns=['target'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# LightGBM parameters with automatic imbalance handling
params = {
    'objective': 'binary',            # Binary classification
    'metric': 'auc',                  # Use AUC as the evaluation metric
    'boosting_type': 'gbdt',          # Gradient Boosting Decision Trees
    'learning_rate': 0.05,            # Learning rate
    'num_leaves': 31,                 # Max leaves in a tree
    'feature_fraction': 0.8,          # Fraction of features for tree building
    'bagging_fraction': 0.8,          # Fraction of data for bagging
    'bagging_freq': 5,                # Perform bagging every 5 iterations
    'is_unbalance': True,             # Automatically handle imbalance
    'seed': 42                        # Random seed for reproducibility
}



# Train LightGBM model with explicit num_boost_round
print("Training model with one-hot encoded data and automatic imbalance handling...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,  # Specify the number of boosting rounds
    valid_sets=[train_data, test_data],
    early_stopping_rounds=50,
    verbose_eval=50
)




# Predict probabilities and evaluate
y_pred_prob = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC: {auc:.4f}")

# Optimize decision threshold for F1-Score
def optimize_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_idx = f1_scores.argmax()
    return thresholds[optimal_idx]

threshold = optimize_threshold(y_test, y_pred_prob)
y_pred_optimized = (y_pred_prob >= threshold).astype(int)
f1_optimized = f1_score(y_test, y_pred_optimized)
print(f"Optimized F1-Score: {f1_optimized:.4f}, Threshold: {threshold:.4f}")

# Summary of Results
print("\nSummary of Results:")
print(f"AUC: {auc:.4f}")
print(f"Optimized F1-Score: {f1_optimized:.4f}, Threshold: {threshold:.4f}")
