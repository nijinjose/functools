import lightgbm as lgb
from optuna.integration import LightGBMPruningCallback

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'scale_pos_weight': (len(y_train) - sum(y_train)) / sum(y_train)
    }

    lgb_train = lgb.Dataset(X_train, y_train)
    cv_results = lgb.cv(
        params,
        lgb_train,
        nfold=5,
        stratified=True,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=False,
        seed=42,
        callbacks=[LightGBMPruningCallback(trial, 'auc-mean')]
    )

    return max(cv_results['auc-mean'])
