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














import joblib
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, confusion_matrix

# 1. Load the saved data
output_dir = "analysis_outputs"
analysis_data = joblib.load(os.path.join(output_dir, "analysis_data.joblib"))

# Extract components
X_train = analysis_data["X_train_full"]
y_train = analysis_data["y_train_full"]
train_predictions = analysis_data["train_predictions"]
X_test = analysis_data["X_test"]
y_test = analysis_data["y_test"]
test_predictions = analysis_data["test_predictions"]
model = analysis_data["final_model"]
cv_scores = analysis_data["cv_auc_scores"]

# 2. Load full datasets with predictions
train_full = pd.read_csv(os.path.join(output_dir, "train_data_with_predictions.csv"))
test_full = pd.read_csv(os.path.join(output_dir, "test_data_with_predictions.csv"))

# 3. Basic Analysis
print("\nModel Performance Metrics:")
print(f"CV AUC Scores: {cv_scores}")
print(f"Average CV AUC: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
print(f"Test AUC: {roc_auc_score(y_test, test_predictions):.4f}")

# 4. Prediction Distribution Analysis
print("\nPrediction Distributions:")
print("\nTrain Predictions:")
print(train_full['prediction'].describe())
print("\nTest Predictions:")
print(test_full['prediction'].describe())

# 5. Prediction Range Analysis
def analyze_predictions(predictions, name=""):
    ranges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dist = pd.cut(predictions, ranges).value_counts().sort_index()
    print(f"\n{name} Predictions by Range (%):")
    print((dist/len(predictions) * 100))

analyze_predictions(train_full['prediction'], "Train")
analyze_predictions(test_full['prediction'], "Test")

# 6. Feature Importance (if using LightGBM)
if hasattr(model, 'feature_importance'):
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importance(),
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))

# 7. Confusion Matrix Analysis (using 0.5 threshold)
def print_confusion_stats(y_true, y_pred, name=""):
    y_pred_binary = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{name} Confusion Matrix Stats:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print(f"Precision: {tp/(tp+fp):.4f}")
    print(f"Recall: {tp/(tp+fn):.4f}")

print_confusion_stats(y_test, test_predictions, "Test Set")

# 8. Prediction vs Actual Analysis
def analyze_accuracy_by_range(y_true, predictions, name=""):
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': predictions
    })
    
    ranges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    df['pred_range'] = pd.cut(df['predicted'], ranges)
    
    analysis = df.groupby('pred_range').agg({
        'actual': ['count', 'mean'],
        'predicted': 'mean'
    })
    
    print(f"\n{name} Accuracy by Prediction Range:")
    print(analysis)

analyze_accuracy_by_range(y_test, test_predictions, "Test Set")








import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# Ensure the output directory exists
output_dir = "analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# ROC Curves for Train and Test
# -----------------------------
# Compute ROC curve and AUC for the training set
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, train_predictions)
roc_auc_train = auc(fpr_train, tpr_train)

# Compute ROC curve and AUC for the test set
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, test_predictions)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC curves for both sets in one figure
plt.figure(figsize=(10, 8))
plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {roc_auc_train:.2f})", color="blue")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {roc_auc_test:.2f})", color="red")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Train and Test Sets")
plt.legend(loc="lower right")
roc_curve_path = os.path.join(output_dir, "roc_curves.png")
plt.savefig(roc_curve_path)
plt.show()

print(f"Train AUC: {roc_auc_train:.4f}")
print(f"Test AUC: {roc_auc_test:.4f}")

# ----------------------------------------------------
# Distribution of Predictions and Average Prediction Values
# ----------------------------------------------------
# Print descriptive statistics for both sets
print("\nTrain Prediction Summary:")
print(train_full['prediction'].describe())

print("\nTest Prediction Summary:")
print(test_full['prediction'].describe())

# Plot prediction distributions for both train and test sets on the same plot
plt.figure(figsize=(10, 8))
sns.histplot(train_full['prediction'], bins=20, kde=True, color="blue", label="Train Predictions", alpha=0.6)
sns.histplot(test_full['prediction'], bins=20, kde=True, color="red", label="Test Predictions", alpha=0.6)
plt.xlabel("Prediction Value")
plt.ylabel("Frequency")
plt.title("Distribution of Predictions for Train and Test Sets")
plt.legend()
dist_plot_path = os.path.join(output_dir, "predictions_distribution.png")
plt.savefig(dist_plot_path)
plt.show()

