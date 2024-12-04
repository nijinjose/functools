# Import necessary libraries
import lightgbm as lgb
import optuna
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib

# Define the CarCancellationPredictor class
class CarCancellationPredictor:
    def __init__(self):
        """
        Initialize the predictor with base parameters for LightGBM.
        """
        self.base_params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'seed': 42,
            'metric': ['auc', 'binary_logloss']
        }
        self.model = None
        self.feature_names = None

    def calculate_pos_weight(self, y):
        """
        Calculate scale_pos_weight for handling class imbalance.

        Parameters:
        y (pd.Series or np.array): The target labels.

        Returns:
        float: The calculated scale_pos_weight.
        """
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        pos_weight = neg_count / pos_count
        return pos_weight

    def objective(self, trial):
        """
        Objective function for Optuna to optimize.

        Parameters:
        trial (optuna.trial.Trial): A trial object for parameter suggestions.

        Returns:
        float: The mean AUC score across all folds.
        """
        max_depth = trial.suggest_int('max_depth', 3, 12)
        params = {
            **self.base_params,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': max_depth,
            'num_leaves': trial.suggest_int('num_leaves', 20, 2 ** max_depth),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 20.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 20.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 0.1)
        }

        # K-Fold Cross-Validation
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

            train_data = lgb.Dataset(X_fold_train, y_fold_train)
            val_data = lgb.Dataset(X_fold_val, y_fold_val)

            try:
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=10000,
                    valid_sets=[val_data],
                    early_stopping_rounds=50,
                    verbose_eval=False  # Suppress training output
                )
            except Exception as e:
                mlflow.log_param('error', str(e))
                raise e

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

            del train_data, val_data  # Free up memory

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        with mlflow.start_run(nested=True):
            mlflow.log_metrics({
                'cv_mean_auc': cv_mean,
                'cv_std_auc': cv_std,
                'cv_stability': cv_std / cv_mean
            })

        return cv_mean

    def train(self, X_train, X_test, y_train, y_test, n_trials=50):
        """
        Train the LightGBM model using Optuna for hyperparameter optimization.

        Parameters:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        y_train (pd.Series or np.array): Training labels.
        y_test (pd.Series or np.array): Test labels.
        n_trials (int): Number of hyperparameter optimization trials.

        Returns:
        tuple: The trained model, evaluation metrics, and feature importance DataFrame.
        """
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.feature_names = X_train.columns.tolist()

        # Note: Assuming mlflow tracking URI and experiment name are set in earlier cells

        with mlflow.start_run():
            # Log dataset info
            pos_weight = self.calculate_pos_weight(y_train)
            mlflow.log_params({
                'n_features': X_train.shape[1],
                'n_samples': X_train.shape[0],
                'pos_class_ratio': y_train.mean(),
                'calculated_pos_weight': pos_weight
            })

            # Set seed for reproducibility
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(self.objective, n_trials=n_trials)

            # Retrieve the best parameters
            best_params = {**self.base_params, **study.best_params}
            best_params['scale_pos_weight'] = pos_weight

            # Train final model on the entire training data
            train_data = lgb.Dataset(X_train, y_train)
            val_data = lgb.Dataset(X_test, y_test)

            self.model = lgb.train(
                best_params,
                train_data,
                num_boost_round=10000,
                valid_sets=[val_data],
                early_stopping_rounds=50,
                verbose_eval=200
            )

            # Evaluate on test set
            test_pred = self.model.predict(X_test)
            test_metrics = {
                'auc': roc_auc_score(y_test, test_pred),
                'average_precision': average_precision_score(y_test, test_pred),
                'brier_score': brier_score_loss(y_test, test_pred)
            }

            # Log results
            mlflow.log_params(study.best_params)
            mlflow.log_metrics(test_metrics)
            mlflow.lightgbm.log_model(self.model, "model")

            # Feature importance
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance_gain': self.model.feature_importance(importance_type='gain'),
                'importance_split': self.model.feature_importance(importance_type='split'),
            })

            # Calculate importance percentages
            total_gain = importance['importance_gain'].sum()
            importance['gain_percentage'] = (importance['importance_gain'] / total_gain) * 100

            total_split = importance['importance_split'].sum()
            importance['split_percentage'] = (importance['importance_split'] / total_split) * 100

            # Sort by importance_gain
            importance = importance.sort_values('importance_gain', ascending=False)

            # Save and log feature importance
            importance.to_csv('feature_importance.csv', index=False)
            mlflow.log_artifact('feature_importance.csv')

            return self.model, test_metrics, importance

    def predict(self, X):
        """
        Predict using the trained LightGBM model.

        Parameters:
        X (pd.DataFrame): Feature set for prediction.

        Returns:
        np.array: Predicted probabilities.
        """
        if self.model is None:
            raise Exception("Model has not been trained yet.")
        return self.model.predict(X)

    def save_model(self, path):
        """
        Save the trained model to a file.

        Parameters:
        path (str): File path to save the model.
        """
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, path)

    def load_model(self, path):
        """
        Load a model from a file.

        Parameters:
        path (str): File path to load the model from.
        """
        saved_data = joblib.load(path)
        self.model = saved_data['model']
        self.feature_names = saved_data['feature_names']

# Usage example
# Assuming df_main (features) and df_target (target) are already defined and preprocessed
if __name__ == "__main__":
    # Split the data into training and test sets
    X = df_main
    y = df_target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Set the MLflow tracking URI and experiment name
    mlflow.set_tracking_uri("https://your-mlflow-server.com")
    mlflow.set_experiment("car_cancellation_prediction")

    # Initialize the predictor
    predictor = CarCancellationPredictor()

    # Train the model
    model, metrics, importance = predictor.train(X_train, X_test, y_train, y_test, n_trials=50)

    # Print evaluation metrics
    print("Test AUC:", metrics['auc'])
    print("Average Precision Score:", metrics['average_precision'])
    print("Brier Score Loss:", metrics['brier_score'])

    # Save the model for future use
    predictor.save_model('car_cancellation_model.pkl')

    # Load the model when needed
    # predictor.load_model('car_cancellation_model.pkl')

    # Make predictions on new data
    # X_new = pd.DataFrame(...)  # Your new data
    # predictions = predictor.predict(X_new)
