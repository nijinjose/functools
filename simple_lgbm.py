import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import mlflow
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns





# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Define preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Apply transformations
X = df.drop('venda', axis=1)
y = df['venda']
X_processed = preprocessor.fit_transform(X)




#OR
#Feature-engine library for advanced feature engineering.
from feature_engine.encoding import OneHotEncoder as fe_OneHotEncoder
from feature_engine.imputation import MeanMedianImputer

# Initialize encoders and imputers
encoder = fe_OneHotEncoder(variables=categorical_cols)
imputer = MeanMedianImputer(imputation_method='median', variables=numerical_cols)

# Apply transformations
X_encoded = encoder.fit_transform(X)
X_imputed = imputer.fit_transform(X_encoded)






# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# Define LightGBM model
lgbm = lgb.LGBMClassifier(objective='binary', is_unbalance=True, random_state=42)

# Train model
lgbm.fit(X_train, y_train)



def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'is_unbalance': True,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    gbm = lgb.train(param, lgb.Dataset(X_train, label=y_train), valid_sets=[lgb.Dataset(X_test, label=y_test)], early_stopping_rounds=10)
    preds = gbm.predict(X_test)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test, pred_labels)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)



from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Predictions
y_pred = lgbm.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC-AUC Score: {roc_auc}')




