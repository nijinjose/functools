import lightgbm as lgb
import optuna
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# -------------------------------------------------
# Data Preparation
# -------------------------------------------------
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

# -------------------------------------------------
# Hyperparameter Tuning via Optuna
# -------------------------------------------------
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
    
    # Log the test AUC for this trial (optional)
    with mlflow.start_run(nested=True):
        mlflow.log_metric("test_auc", auc)
    
    return auc

# Set up the Optuna study and MLflow experiment
study = optuna.create_study(direction="maximize")
mlflow.set_experiment("client_approach_lightgbm_experiment")

with mlflow.start_run(run_name="optuna_client_approach_tuning"):
    study.optimize(objective, n_trials=50)
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_test_auc", study.best_value)
    print(f"Best Test AUC: {study.best_value}")
    print(f"Best Parameters: {study.best_params}")

# -------------------------------------------------
# Final Model Training and Artifact Logging
# -------------------------------------------------
# Prepare the final hyperparameters by merging in any fixed values.
final_params = study.best_params.copy()
final_params.update({
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'seed': 42
})

# Retrain the final model using the entire training set.
final_model = lgb.train(
    final_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],  # Still using test_data for early stopping if desired
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    verbose_eval=False
)

# Evaluate the final model on the test set.
final_preds = final_model.predict(X_test)
final_auc = roc_auc_score(y_test, final_preds)
print(f"Final Test AUC: {final_auc}")

# Log the final model as an MLflow artifact.
mlflow.lightgbm.log_model(final_model, "final_model")
