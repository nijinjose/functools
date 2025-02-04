import os
import joblib

# ---------------------------
# Post-Training: Save Data and Predictions for Analysis
# ---------------------------

# Create an output directory to store the files
output_dir = "analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# Generate predictions on the full training set (if desired)
train_predictions = final_model.predict(X_train_full)

# Save training data with predictions to a CSV file
train_df = X_train_full.copy()
train_df['true_target'] = y_train_full
train_df['prediction'] = train_predictions
train_csv_path = os.path.join(output_dir, "train_data_with_predictions.csv")
train_df.to_csv(train_csv_path, index=False)
print(f"Training data with predictions saved to {train_csv_path}")

# Save test data with predictions to a CSV file
test_df = X_test.copy()
test_df['true_target'] = y_test
test_df['prediction'] = test_predictions
test_csv_path = os.path.join(output_dir, "test_data_with_predictions.csv")
test_df.to_csv(test_csv_path, index=False)
print(f"Test data with predictions saved to {test_csv_path}")

# Optionally, save the raw feature splits and targets if needed
X_train_csv = os.path.join(output_dir, "X_train_full.csv")
y_train_csv = os.path.join(output_dir, "y_train_full.csv")
X_test_csv = os.path.join(output_dir, "X_test.csv")
y_test_csv = os.path.join(output_dir, "y_test.csv")

X_train_full.to_csv(X_train_csv, index=False)
y_train_full.to_csv(y_train_csv, index=False)
X_test.to_csv(X_test_csv, index=False)
y_test.to_csv(y_test_csv, index=False)

print(f"Raw splits saved: {X_train_csv}, {y_train_csv}, {X_test_csv}, {y_test_csv}")

# Save all objects in a dictionary as a joblib file for easy reloading in the future
analysis_data = {
    "X_train_full": X_train_full,
    "y_train_full": y_train_full,
    "train_predictions": train_predictions,
    "X_test": X_test,
    "y_test": y_test,
    "test_predictions": test_predictions,
    "final_model": final_model,
    "params": params,
    "cv_auc_scores": auc_scores,   # from cross-validation
    "average_cv_auc": avg_auc,
    "test_auc": test_auc
}

joblib_path = os.path.join(output_dir, "analysis_data.joblib")
joblib.dump(analysis_data, joblib_path)
print(f"Analysis data saved as joblib file to {joblib_path}")
