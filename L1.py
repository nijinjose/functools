# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For feature selection and dimensionality reduction
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# For model training and evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, classification_report
import lightgbm as lgb

# For hyperparameter tuning
import optuna
from optuna.integration import LightGBMPruningCallback

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# Display the first few rows of the dataset
df.head()


# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])


# # Handle missing values
# # For numerical columns, fill missing values with the median
# numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
# df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# # For categorical columns, fill missing values with the mode
# categorical_cols = df.select_dtypes(include=['object', 'category']).columns
# df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])


# Verify that there are no more missing values
print("Any missing values left:", df.isnull().values.any())


# Check for duplicated rows
duplicate_rows = df[df.duplicated()]
print(f"Number of duplicate rows: {duplicate_rows.shape[0]}")


# Remove duplicate rows if any
df = df.drop_duplicates()

# # Convert data types if necessary
# # For example, ensure that categorical columns are of type 'category'
# for col in categorical_cols:
#     df[col] = df[col].astype('category')

# Summary statistics for numerical columns
df[numerical_cols].describe()


# Distribution of the target variable
target_var = 'target'  
print(df[target_var].value_counts(normalize=True))

sns.countplot(x=target_var, data=df)
plt.title('Distribution of Target Variable')
plt.show()



# Compute correlation matrix for numerical features
corr_matrix = df[numerical_cols].corr()

# Plot heatmap of correlations
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# Identify highly correlated features (absolute correlation > 0.8)
high_corr_var = np.where(np.abs(corr_matrix) > 0.8)
high_corr_var = [(numerical_cols[x], numerical_cols[y]) for x, y in zip(*high_corr_var) if x != y and x < y]

print("Highly correlated features:")
for var1, var2 in high_corr_var:
    print(f"{var1} and {var2}")


#Feature Importance 
# Encode categorical variables for LightGBM
for col in categorical_cols:
    df[col] = df[col].cat.codes

# Separate features and target
X = df.drop(columns=[target_var])
y = df[target_var]

# Split into training and validation sets
X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Create a basic LightGBM dataset
lgb_train_fs = lgb.Dataset(X_train_fs, y_train_fs)
lgb_eval_fs = lgb.Dataset(X_val_fs, y_val_fs, reference=lgb_train_fs)

# Train a basic LightGBM model to get feature importances
params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'seed': 42
}

model_fs = lgb.train(params, lgb_train_fs, valid_sets=[lgb_eval_fs], num_boost_round=100, early_stopping_rounds=10, verbose_eval=False)

# Get feature importances
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model_fs.feature_importance()
})

# Plot top 20 important features
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
plt.title('Top 20 Feature Importances from LightGBM')
plt.show()




# Remove features with near-zero variance
selector = VarianceThreshold(threshold=0.01)  # Adjust threshold as needed
selector.fit(X)
X_reduced = X[X.columns[selector.get_support(indices=True)]]
print(f"Reduced features from {X.shape[1]} to {X_reduced.shape[1]} after removing low variance features.")


# Use feature importances from LightGBM to select top features
top_features = feature_importances.head(50)['feature'].tolist()  # Select top 50 features
X_selected = X_reduced[top_features]
print(f"Selected top {len(top_features)} features based on importance.")





# Modified run

# Final features and target
X_final = X_selected.copy()
y_final = y.copy()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, stratify=y_final, random_state=42)



# Calculate class weights
from collections import Counter

counter = Counter(y_train)
majority_class = max(counter, key=counter.get)
minority_class = min(counter, key=counter.get)
scale_pos_weight = counter[majority_class] / counter[minority_class]
print(f"Scale_pos_weight: {scale_pos_weight}")

def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': 42,
        'scale_pos_weight': scale_pos_weight,
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)

        gbm = lgb.train(
            param,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_valid],
            early_stopping_rounds=50,
            verbose_eval=False,
            callbacks=[LightGBMPruningCallback(trial, 'auc')]
        )

        preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        auc = roc_auc_score(y_val, preds)
        auc_scores.append(auc)

    return np.mean(auc_scores)


# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('  AUC: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))




# Use best parameters to train final model
best_params = trial.params
best_params['objective'] = 'binary'
best_params['metric'] = 'auc'
best_params['verbosity'] = -1
best_params['boosting_type'] = 'gbdt'
best_params['seed'] = 42
best_params['scale_pos_weight'] = scale_pos_weight

lgb_train_final = lgb.Dataset(X_train, y_train)
lgb_valid_final = lgb.Dataset(X_test, y_test, reference=lgb_train_final)

model = lgb.train(
    best_params,
    lgb_train_final,
    num_boost_round=1000,
    valid_sets=[lgb_valid_final],
    early_stopping_rounds=50,
    verbose_eval=50
)





# threshold values

# Predict on test set
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba >= 0.5).astype(int)  # Default threshold of 0.5

# Evaluate model
auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test AUC: {auc:.4f}')

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * recall * precision / (recall + precision)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f'Optimal Threshold (based on F1 Score): {optimal_threshold}')

# Recompute predictions with optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_optimal))



# Plot feature importances
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importance()
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=importance_df.head(20))
plt.title('Feature Importances from Final Model')
plt.show()



import shap

# Initialize the explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

