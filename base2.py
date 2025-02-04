import lightgbm as lgb
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

# Assume 'df' is your DataFrame with one-hot encoded features and a 'target' column
X = df.drop(columns=['target'])
y = df['target']

# Create a hold-out test set first
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Define fixed (non-tuned) hyperparameters with imbalance handling turned off
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,      # Fixed learning rate
    'num_leaves': 31,          # Default value for num_leaves
    'max_depth': 5,            # Fixed maximum depth
    'min_data_in_leaf': 20,    # Minimum data in a leaf node
    'is_unbalance': False,     # Do not adjust for class imbalance automatically
    'seed': 42
}

# Set the MLflow experiment name
mlflow.set_experiment("non_tuned_lgb_with_holdout")

# Start an overall MLflow run
with mlflow.start_run(run_name="5_fold_cv_and_final_model_with_holdout") as run:
    # Log the fixed hyperparameters and dataset split info
    mlflow.log_params({
        **params,
        'test_size': 0.2,
        'training_samples': len(X_train_full),
        'test_samples': len(X_test),
        'positive_rate_train': y_train_full.mean(),
        'positive_rate_test': y_test.mean()
    })
    
    # Set up a non-stratified 5-fold cross-validation using KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    # Iterate over each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full), start=1):
        print(f"Processing Fold {fold}")
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        # Prepare LightGBM Datasets for training and validation
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        # Train the model on the current fold
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,          # Fixed number of iterations
            valid_sets=[val_data],
            early_stopping_rounds=10,      # Stop if no improvement for 10 rounds
            verbose_eval=False
        )
        
        # Predict on the validation set using the best iteration
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val, y_pred)
        print(f"Fold {fold} ROC AUC: {auc:.4f}")
        auc_scores.append(auc)
        
        # Log fold-specific metrics in a nested MLflow run
        with mlflow.start_run(nested=True, run_name=f"fold_{fold}"):
            mlflow.log_metrics({
                'fold_auc': auc,
                'best_iteration': model.best_iteration,
                'fold_positive_rate': y_val.mean()
            })
    
    # Compute and log the average ROC AUC across the 5 folds
    avg_auc = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    mlflow.log_metrics({
        'average_cv_auc': avg_auc,
        'cv_auc_std': auc_std
    })
    print(f"Average ROC AUC across 5 folds: {avg_auc:.4f} (±{auc_std:.4f})")
    
    # ---------------------
    # Train the Final Model
    # ---------------------
    # Create LightGBM Datasets using the full training data and test set
    final_train_data = lgb.Dataset(X_train_full, label=y_train_full)
    test_data = lgb.Dataset(X_test, label=y_test, reference=final_train_data)
    
    # Train the final model on the full training dataset
    final_model = lgb.train(
        params,
        final_train_data,
        num_boost_round=100,   # Fixed number of iterations
        valid_sets=[test_data],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Evaluate on the hold-out test set
    test_predictions = final_model.predict(X_test)
    test_auc = roc_auc_score(y_test, test_predictions)
    
    # Calculate prediction distribution statistics
    pred_stats = {
        'mean_prediction': test_predictions.mean(),
        'median_prediction': np.median(test_predictions),
        'std_prediction': test_predictions.std(),
        'min_prediction': test_predictions.min(),
        'max_prediction': test_predictions.max()
    }
    
    # Calculate prediction distribution by ranges
    ranges = np.arange(0, 1.1, 0.1)
    pred_dist = pd.cut(test_predictions, ranges).value_counts().sort_index()
    pred_dist_pct = (pred_dist / len(test_predictions)) * 100
    
    # Log final metrics
    mlflow.log_metrics({
        'test_set_auc': test_auc,
        'test_set_mean_pred': pred_stats['mean_prediction'],
        'test_set_median_pred': pred_stats['median_prediction'],
        'test_set_std_pred': pred_stats['std_prediction']
    })
    
    # Log prediction distribution as a JSON artifact
    dist_dict = {f"{int(left*100)}-{int(right*100)}%": val 
                 for (left, right), val in pred_dist_pct.items()}
    with open('prediction_distribution.json', 'w') as f:
        json.dump(dist_dict, f, indent=2)
    mlflow.log_artifact('prediction_distribution.json')
    
    # Log the final model to MLflow
    mlflow.lightgbm.log_model(final_model, artifact_path="final_model")
    print(f"Final model trained and saved to MLflow. Test AUC: {test_auc:.4f}")
    
    # Print prediction distribution
    print("\nPrediction Distribution on Test Set:")
    for range_label, percentage in dist_dict.items():
        print(f"{range_label}: {percentage:.2f}%")
