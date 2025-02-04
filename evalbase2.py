import joblib
import pandas as pd
import numpy as np

# 1. Get predictions
train_predictions = final_model.predict(X_train_full)
test_predictions = final_model.predict(X_test)

# 2. Create results DataFrames - with explicit flattening
train_results_df = pd.DataFrame({
    'actual': y_train_full.flatten() if isinstance(y_train_full, np.ndarray) else y_train_full,
    'predicted': train_predictions.flatten() if isinstance(train_predictions, np.ndarray) else train_predictions
})

test_results_df = pd.DataFrame({
    'actual': y_test.flatten() if isinstance(y_test, np.ndarray) else y_test,
    'predicted': test_predictions.flatten() if isinstance(test_predictions, np.ndarray) else test_predictions
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
    dist = pd.cut(predictions.flatten() if isinstance(predictions, np.ndarray) else predictions, 
                 ranges).value_counts().sort_index()
    return (dist/len(predictions)) * 100

print("\nTrain Predictions Range Distribution (%):")
print(get_prediction_ranges(train_predictions))

print("\nTest Predictions Range Distribution (%):")
print(get_prediction_ranges(test_predictions))
