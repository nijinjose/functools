import lightgbm as lgb
import optuna
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib

class CarCancellationPredictor:
    def __init__(self):
        # Base parameters now use LightGBM's built-in imbalance handling
        self.base_params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'metric': ['auc', 'binary_logloss'],
            'verbose': -1,
            'seed': 42,
            'is_unbalance': True  # Let LightGBM handle imbalance automatically
        }
        self.feature_names = None
        self.model = None

    def objective(self, trial):
        # Suggest hyperparameters with adjusted ranges
        max_depth = trial.suggest_int('max_depth', 3, 10)
        num_leaves = trial.suggest_int('num_leaves', 20, max(2 ** max_depth, 20))
        params = {
            **self.base_params,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 0.1)
        }

        # 5-fold stratified CV for robust performance estimation
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            train_data = lgb.Dataset(X_fold_train, y_fold_train, feature_name=self.feature_names)
            val_data = lgb.Dataset(X_fold_val, y_fold_val, feature_name=self.feature_names)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=10000,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=200)
                ]
            )

            y_pred = model.predict(X_fold_val)
            fold_score = roc_auc_score(y_fold_val, y_pred)
            cv_scores.append(fold_score)

            # Log per-fold metrics
            with mlflow.start_run(nested=True):
                mlflow.log_metrics({
                    f'fold_{fold_idx}_auc': fold_score,
                    f'fold_{fold_idx}_best_iteration': model.best_iteration
                })
                mlflow.set_tag('fold_index', fold_idx)

            # Clear memory for next fold
            del train_data, val_data

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        # Log overall CV results
        with mlflow.start_run(nested=True):
            mlflow.log_metrics({
                'cv_mean_auc': cv_mean,
                'cv_std_auc': cv_std,
                'cv_stability': cv_std / cv_mean
            })

        return cv_mean

    def train(self, X_train, X_test, y_train, y_test, n_trials=50):
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.feature_names = X_train.columns.tolist()

        # Set up MLflow experiment; update the URI and experiment name as needed.
        mlflow.set_tracking_uri("https://your-azure-mlflow-server.com")
        mlflow.set_experiment("car_cancellation_prediction_unbiased")

        with mlflow.start_run():
            mlflow.log_params({
                'n_features': X_train.shape[1],
                'n_samples': X_train.shape[0],
                'pos_class_ratio': y_train.mean(),
            })

            # Initialize an Optuna study for hyperparameter tuning
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(self.objective, n_trials=n_trials)  # Run single-threaded

            # Final training using a fresh train/validation split
            X_train_new, X_val, y_train_new, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )

            best_params = {**self.base_params, **study.best_params}

            train_data = lgb.Dataset(X_train_new, y_train_new, feature_name=self.feature_names)
            val_data = lgb.Dataset(X_val, y_val, feature_name=self.feature_names)

            self.model = lgb.train(
                best_params,
                train_data,
                num_boost_round=10000,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=200)
                ]
            )

            # Evaluate the final model on the test set
            test_pred = self.model.predict(X_test)
            test_metrics = {
                'auc': roc_auc_score(y_test, test_pred),
                'avg_precision': average_precision_score(y_test, test_pred),
                'brier_score': brier_score_loss(y_test, test_pred)
            }

            # Log final best parameters and metrics
            mlflow.log_params(study.best_params)
            mlflow.log_metrics(test_metrics)
            mlflow.lightgbm.log_model(self.model, "model")

            # Log feature importance
            importance_values = self.model.feature_importance()
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_values,
                'importance_percentage': importance_values / sum(importance_values) * 100
            }).sort_values('importance', ascending=False)

            importance.to_csv('feature_importance.csv', index=False)
            mlflow.log_artifact('feature_importance.csv')

            return self.model, test_metrics, importance

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, path)

    def load_model(self, path):
        saved_data = joblib.load(path)
        self.model = saved_data['model']
        self.feature_names = saved_data['feature_names']

# --------------------------------------
# Usage Example:
# --------------------------------------
# Assume `df_main` is your DataFrame containing all features,
# and `df_target` is your Series (or DataFrame) containing the binary target (0/1).

X = df_main
y = df_target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

predictor = CarCancellationPredictor()
model, metrics, importance = predictor.train(X_train, X_test, y_train, y_test, n_trials=50)
