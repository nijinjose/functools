import lightgbm as lgb
import optuna
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

# Assuming 'df' is your DataFrame with one-hot encoded features and a 'target' column
X = df.drop(columns=['target'])
y = df['target']

# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create LightGBM Dataset Objects
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Define the Objective Function for Optuna
def objective(trial):
    # Hyperparameter Search Space
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
        'seed': 42,
        'is_unbalance': True
    }

    # Train the LightGBM Model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data, train_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False)
        ]
    )

    # Predict on the Test Set
    y_pred_prob = model.predict(X_test)

    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_prob)

    # Log Parameters and Metrics to MLflow
    with mlflow.start_run(nested=True):  # Nested runs for each trial
        mlflow.log_params(params)
        mlflow.log_metric("auc", auc)

    return auc

# Optuna Study for Hyperparameter Tuning
study = optuna.create_study(direction="maximize")
mlflow.set_experiment("lightgbm_optuna_experiment")  # Set MLflow Experiment Name

with mlflow.start_run(run_name="optuna_tuning"):
    study.optimize(objective, n_trials=50)  # Run 50 trials

    # Log Best Trial to MLflow
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_auc", study.best_value)

    print(f"Best AUC: {study.best_value}")
    print(f"Best Parameters: {study.best_params}")
