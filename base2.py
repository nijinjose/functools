import lightgbm as lgb
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

# Create a hold-out test set first
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Define fixed parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': 5,
    'min_data_in_leaf': 20,
    'is_unbalance': False,
    'seed': 42
}

# Set up MLflow
mlflow.set_experiment("non_tuned_lgb_with_holdout")

with mlflow.start_run(run_name="5_fold_cv_and_final_model"):
    # Log basic parameters
    mlflow.log_params(params)
    
    # Set up 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    # Cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full), start=1):
        print(f"Processing Fold {fold}")
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)
        print(f"Fold {fold} ROC AUC: {auc:.4f}")
        auc_scores.append(auc)
    
    # Log average CV performance
    avg_auc = np.mean(auc_scores)
    print(f"Average ROC AUC across 5 folds: {avg_auc:.4f}")
    mlflow.log_metric('average_cv_auc', avg_auc)
    
    # Train final model
    final_train_data = lgb.Dataset(X_train_full, label=y_train_full)
    test_data = lgb.Dataset(X_test, label=y_test)
    
    final_model = lgb.train(
        params,
        final_train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Evaluate final model
    test_predictions = final_model.predict(X_test)
    test_auc = roc_auc_score(y_test, test_predictions)
    print(f"Final Test AUC: {test_auc:.4f}")
    
    # Log final model and test performance
    mlflow.log_metric('test_auc', test_auc)
    mlflow.lightgbm.log_model(final_model, "final_model")
