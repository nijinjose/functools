import lightgbm as lgb
import optuna
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Assume df is your DataFrame with one-hot encoded features and a 'target' column
X = df.drop(columns=['target'])
y = df['target']

def objective(trial):
    # Define hyperparameter search space WITHOUT the imbalance parameter.
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'seed': 42  # No imbalance parameter here
    }

    # Setup Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False)
            ],
            verbose_eval=False
        )

        preds = model.predict(X_valid)
        auc = roc_auc_score(y_valid, preds)
        auc_scores.append(auc)

    # Log mean CV AUC to MLflow (optional)
    cv_mean_auc = np.mean(auc_scores)
    with mlflow.start_run(nested=True):
        mlflow.log_metric("cv_mean_auc", cv_mean_auc)

    return cv_mean_auc

# Initialize and run the Optuna study.
study = optuna.create_study(direction="maximize")
mlflow.set_experiment("lightgbm_no_imbalance_cv_experiment")

with mlflow.start_run(run_name="optuna_no_imbalance_cv_tuning"):
    study.optimize(objective, n_trials=50)
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_cv_auc", study.best_value)

    print(f"Best CV AUC: {study.best_value}")
    print(f"Best Parameters: {study.best_params}")






-------------------------OR -------------if Cross validation is used ----------------------




import lightgbm as lgb
import optuna
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Assume:
# - `df` is a DataFrame containing only feature columns.
# - `df_target` is a separate Series (or DataFrame) containing the target labels.
X = df
y = df_target

# Split the data into training and test sets, preserving class proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create LightGBM Datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

def objective(trial):
    # Define the hyperparameter search space (without explicit imbalance handling)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'seed': 42
    }
    
    # Train LightGBM using the training set, evaluating on the test set (as a validation set)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        verbose_eval=False
    )
    
    # Predict on the test set and calculate AUC
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    
    # Log the metric (optional)
    with mlflow.start_run(nested=True):
        mlflow.log_metric("test_auc", auc)
    
    return auc

# Set up an Optuna study and MLflow experiment
study = optuna.create_study(direction="maximize")
mlflow.set_experiment("client_approach_lightgbm_experiment")

with mlflow.start_run(run_name="optuna_client_approach_tuning"):
    study.optimize(objective, n_trials=50)
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_test_auc", study.best_value)

    print(f"Best Test AUC: {study.best_value}")
    print(f"Best Parameters: {study.best_params}")

