# After your model training, add these lines to save artifacts:

# Save training and test data
train_test_data = {
    'X_train': X_train_full,
    'X_test': X_test,
    'y_train': y_train_full,
    'y_test': y_test
}

# Get predictions for both train and test
train_predictions = final_model.predict(X_train_full)
test_predictions = final_model.predict(X_test)

# Create DataFrames with actual values and predictions
train_results_df = pd.DataFrame({
    'actual': y_train_full,
    'predicted': train_predictions
})

test_results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': test_predictions
})

# Save all artifacts
import joblib

# Save model and data
joblib.dump(final_model, 'final_model.joblib')
joblib.dump(train_test_data, 'train_test_data.joblib')

# Save predictions as csv
train_results_df.to_csv('train_predictions.csv', index=False)
test_results_df.to_csv('test_predictions.csv', index=False)

# Let's analyze the distribution of predictions
print("\nTrain Predictions Distribution:")
print(train_results_df['predicted'].describe())

print("\nTest Predictions Distribution:")
print(test_results_df['predicted'].describe())

# Let's see prediction ranges distribution
def get_prediction_ranges(predictions):
    ranges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dist = pd.cut(predictions, ranges).value_counts().sort_index()
    return (dist/len(predictions)) * 100

print("\nTrain Predictions Range Distribution (%):")
print(get_prediction_ranges(train_predictions))

print("\nTest Predictions Range Distribution (%):")
print(get_prediction_ranges(test_predictions))
