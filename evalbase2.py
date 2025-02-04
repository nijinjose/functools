import joblib
import pandas as pd
import numpy as np

# 1. Get predictions
train_predictions = final_model.predict(X_train_full)
test_predictions = final_model.predict(X_test)

# 2. Create results DataFrames
# Ensure targets are 1-dimensional
y_train_1d = y_train_full.ravel() if hasattr(y_train_full, 'ravel') else y_train_full
y_test_1d = y_test.ravel() if hasattr(y_test, 'ravel') else y_test

train_results_df = pd.DataFrame({
    'actual': y_train_1d,
    'predicted': train_predictions
})

test_results_df = pd.DataFrame({
    'actual': y_test_1d,
    'predicted': test_predictions
})

# 3. Save everything
joblib.dump(final_model, 'final_model.joblib')
train_results_df.to_csv('train_predictions.csv', index=False)
test_results_df.to_csv('test_predictions.csv', index=False)

# 4. Analyze predictions
print("\nTrain Predictions Distribution:")
print(train_results_df['predicted'].describe())

print("\nTest Predictions Distribution:")
print(test_results_df['predicted'].describe())

# 5. Show prediction ranges distribution
def get_prediction_ranges(predictions):
    ranges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dist = pd.cut(predictions, ranges).value_counts().sort_index()
    return (dist/len(predictions)) * 100

print("\nTrain Predictions Range Distribution (%):")
print(get_prediction_ranges(train_predictions))

print("\nTest Predictions Range Distribution (%):")
print(get_prediction_ranges(test_predictions))
