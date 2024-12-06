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
        self.base_params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'seed': 42,
            'metric': ['auc', 'binary_logloss']
        }
        self.feature_names = None
        self.model = None

    def calculate_pos_weight(self, y_train):
        """Calculate balanced class weight."""
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        pos_weight = neg_count / pos_count
        return pos_weight

    def objective(self, trial):
        # Suggest hyperparameters
        max_depth = trial.suggest_int('max_depth', 3, 12)
        # Ensure num_leaves range is always valid
        num_leaves_upper = max(20, 2 ** max_depth)

        params = {
            **self.base_params,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': max_depth,
            'num_leaves': trial.suggest_int('num_leaves', 20, num_leaves_upper),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 20.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 20.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 0.1)
        }

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            # Calculate fold-specific pos_weight
            fold_pos_weight = self.calculate_pos_weight(y_fold_train)
            params['scale_pos_weight'] = fold_pos_weight

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

            with mlflow.start_run(nested=True):
                mlflow.log_metrics({
                    f'fold_{fold_idx}_auc': fold_score,
                    f'fold_{fold_idx}_best_iteration': model.best_iteration,
                    f'fold_{fold_idx}_pos_weight': fold_pos_weight
                })
                mlflow.set_tag('fold_index', fold_idx)

            # Free memory for next fold
            del train_data, val_data

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        # Log CV results
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

        # Set up MLflow experiment
        mlflow.set_tracking_uri("https://your-azure-mlflow-server.com")
        mlflow.set_experiment("car_cancellation_prediction")

        with mlflow.start_run():
            # Log dataset info
            pos_weight = self.calculate_pos_weight(y_train)
            mlflow.log_params({
                'n_features': X_train.shape[1],
                'n_samples': X_train.shape[0],
                'pos_class_ratio': y_train.mean(),
                'calculated_pos_weight': pos_weight
            })

            # Initialize Optuna study
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(self.objective, n_trials=n_trials)  # single-threaded

            # Final training with best params on a fresh train/val split
            X_train_new, X_val, y_train_new, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )

            best_params = {**self.base_params, **study.best_params}
            best_params['scale_pos_weight'] = self.calculate_pos_weight(y_train_new)

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

            # Evaluate on test set
            test_pred = self.model.predict(X_test)
            test_metrics = {
                'auc': roc_auc_score(y_test, test_pred),
                'avg_precision': average_precision_score(y_test, test_pred),
                'brier_score': brier_score_loss(y_test, test_pred)
            }

            # Log final model and metrics
            mlflow.log_params(study.best_params)
            mlflow.log_metrics(test_metrics)
            mlflow.lightgbm.log_model(self.model, "model")

            # Feature importance
            importance_values = self.model.feature_importance()
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_values,
                'importance_percentage': importance_values / sum(importance_values) * 100
            }).sort_values('importance', ascending=False)

            # Save and log feature importance
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


# Usage Example:
# df_main: DataFrame with all features
# df_target: Series with binary labels (0/1)
# Ensure df_main and df_target are defined.

X = df_main
y = df_target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

predictor = CarCancellationPredictor()
model, metrics, importance = predictor.train(X_train, X_test, y_train, y_test, n_trials=50)
