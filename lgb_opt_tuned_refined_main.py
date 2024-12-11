import lightgbm as lgb
import optuna
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib
import logging
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    auc: float
    avg_precision: float
    brier_score: float
    feature_importance: pd.DataFrame

class CarCancellationPredictor:
    """
    A predictor for car cancellation using LightGBM with automated hyperparameter optimization.
    
    This class assumes MLflow has been configured externally in the notebook:
    ```python
    mlflow.set_tracking_uri("your_mlflow_uri")
    mlflow.set_experiment("your_experiment_name")
    ```
    """
    
    def __init__(self):
        """Initialize predictor with base LightGBM parameters."""
        self.base_params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'seed': 42,
            'metric': ['auc', 'binary_logloss']
        }
        self.feature_names = None
        self.model = None
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging with both file and console output."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("model_training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _calculate_pos_weight(self, y_train: pd.Series) -> float:
        """Calculate balanced class weight for imbalanced datasets."""
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        return neg_count / pos_count if pos_count > 0 else 1.0

    def _create_optuna_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Generate hyperparameter suggestions for the current trial."""
        max_depth = trial.suggest_int('max_depth', 3, 12)
        num_leaves_upper = max(20, 2 ** max_depth)
        
        return {
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

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization."""
        params = self._create_optuna_parameters(trial)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        # Get current MLflow run
        run = mlflow.active_run()
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X_train, self.y_train)):
            # Prepare fold data
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            params['scale_pos_weight'] = self._calculate_pos_weight(y_fold_train)

            train_data = lgb.Dataset(X_fold_train, y_fold_train, feature_name=self.feature_names)
            val_data = lgb.Dataset(X_fold_val, y_fold_val, feature_name=self.feature_names)

            try:
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

                # Log metrics with fold prefix
                mlflow.log_metrics({
                    f'fold_{fold_idx}/auc': fold_score,
                    f'fold_{fold_idx}/best_iteration': model.best_iteration,
                    f'fold_{fold_idx}/pos_weight': params['scale_pos_weight']
                }, step=fold_idx)

            finally:
                del train_data, val_data

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        # Log CV summary metrics
        mlflow.log_metrics({
            'cv/mean_auc': cv_mean,
            'cv/std_auc': cv_std,
            'cv/stability': cv_std / cv_mean
        })

        return cv_mean

    def train(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series, 
        n_trials: int = 50
    ) -> Tuple[lgb.Booster, ModelMetrics]:
        """
        Train the model with hyperparameter optimization.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            n_trials: Number of optimization trials
            
        Returns:
            Tuple containing trained model and evaluation metrics
        """
        self.logger.info("Starting model training process")
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.feature_names = X_train.columns.tolist()

        with mlflow.start_run() as run:
            # Log training metadata
            mlflow.log_params({
                'n_features': X_train.shape[1],
                'n_samples': X_train.shape[0],
                'pos_class_ratio': y_train.mean(),
                'n_trials': n_trials,
                'training_start': pd.Timestamp.now().isoformat()
            })

            # Hyperparameter optimization
            self.logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(self._objective, n_trials=n_trials)

            # Final model training
            self.logger.info("Training final model with best parameters")
            X_train_new, X_val, y_train_new, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )

            best_params = {**self.base_params, **study.best_params}
            best_params['scale_pos_weight'] = self._calculate_pos_weight(y_train_new)

            # Log best parameters
            mlflow.log_params({f'best_{k}': v for k, v in study.best_params.items()})

            train_data = lgb.Dataset(X_train_new, y_train_new, feature_name=self.feature_names)
            val_data = lgb.Dataset(X_val, y_val, feature_name=self.feature_names)

            try:
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

                # Calculate and log final metrics
                test_pred = self.model.predict(X_test)
                metrics = ModelMetrics(
                    auc=roc_auc_score(y_test, test_pred),
                    avg_precision=average_precision_score(y_test, test_pred),
                    brier_score=brier_score_loss(y_test, test_pred),
                    feature_importance=self._calculate_feature_importance()
                )

                mlflow.log_metrics({
                    'final/test_auc': metrics.auc,
                    'final/test_avg_precision': metrics.avg_precision,
                    'final/test_brier_score': metrics.brier_score
                })

                # Log feature importance
                importance_path = 'feature_importance.csv'
                metrics.feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                
                # Log the model
                mlflow.lightgbm.log_model(self.model, "model")

                self.logger.info(f"Training complete. Test AUC: {metrics.auc:.4f}")
                return self.model, metrics

            finally:
                del train_data, val_data

    def _calculate_feature_importance(self) -> pd.DataFrame:
        """Calculate and format feature importance scores."""
        importance_values = self.model.feature_importance()
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_values,
            'importance_percentage': importance_values / sum(importance_values) * 100
        }).sort_values('importance', ascending=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)

    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load a saved model from disk."""
        saved_data = joblib.load(path)
        self.model = saved_data['model']
        self.feature_names = saved_data['feature_names']
        self.logger.info(f"Model loaded from {path}")
