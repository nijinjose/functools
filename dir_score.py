import joblib
import lightgbm as lgb

# Hard-coded paths for LightGBM models
lgb_models = {
    "a0": "C:/Users/your_user/Documents/results/a0/model_a0.lgb",
    "b1": "C:/Users/your_user/Documents/results/b1/model_b1.lgb",
    "b2": "C:/Users/your_user/Documents/results/b2/model_b2.lgb",
    "c3": "C:/Users/your_user/Documents/results/c3/model_c3.lgb",
}

# Hard-coded paths for joblib models
joblib_models = {
    "a0": "C:/Users/your_user/Documents/results/a0/joblib/model_a0.joblib",
    "b1": "C:/Users/your_user/Documents/results/b1/joblib/model_b1.joblib",
    "b2": "C:/Users/your_user/Documents/results/b2/joblib/model_b2.joblib",
    "c3": "C:/Users/your_user/Documents/results/c3/joblib/model_c3.joblib",
}

# Load LightGBM models
loaded_lgb_models = {}
for name, path in lgb_models.items():
    print(f"Loading LightGBM model: {name} from {path}")
    try:
        loaded_lgb_models[name] = lgb.Booster(model_file=path)
        print(f"Successfully loaded {name}")
    except Exception as e:
        print(f"Failed to load LightGBM model {name}: {e}")

# Load joblib models
loaded_joblib_models = {}
for name, path in joblib_models.items():
    print(f"Loading joblib model: {name} from {path}")
    try:
        loaded_joblib_models[name] = joblib.load(path)
        print(f"Successfully loaded {name}")
    except Exception as e:
        print(f"Failed to load joblib model {name}: {e}")






import pandas as pd

# Assuming df_test is your test DataFrame
df_test = pd.read_csv("test_data.csv")  # Replace with the actual path to your test data

# Create a DataFrame to store predictions
predictions = pd.DataFrame()
predictions["index"] = df_test.index  # Add index column for reference

# Score with LightGBM models
for model_name, model in lgb_models.items():
    print(f"Scoring with LightGBM model: {model_name}")
    try:
        predictions[model_name] = model.predict(df_test)  # Use predict for LightGBM
    except Exception as e:
        print(f"Failed to score with model {model_name}: {e}")

# Score with joblib models
for model_name, model in joblib_models.items():
    print(f"Scoring with joblib model: {model_name}")
    try:
        predictions[model_name] = model.predict(df_test)  # Use predict for joblib models
    except Exception as e:
        print(f"Failed to score with model {model_name}: {e}")

# Save predictions to a CSV file
predictions.to_csv("model_predictions.csv", index=False)
print("Predictions saved to 'model_predictions.csv'")


# Display loaded models
print("Loaded LightGBM models:", list(loaded_lgb_models.keys()))
print("Loaded joblib models:", list(loaded_joblib_models.keys()))






import os
import pandas as pd
import lightgbm as lgb
import joblib

# Define your test DataFrame
df_test = pd.read_csv("test_data.csv")  # Update with the correct test data file path

# Paths for LightGBM and Joblib models
lgb_models = {
    "a0": "C:/Users/your_user/Documents/results/a0/model_a0.lgb",
    "b1": "C:/Users/your_user/Documents/results/b1/model_b1.lgb",
    "b2": "C:/Users/your_user/Documents/results/b2/model_b2.lgb",
    "c3": "C:/Users/your_user/Documents/results/c3/model_c3.lgb",
}

joblib_models = {
    "a0": "C:/Users/your_user/Documents/results/a0/joblib/model_a0.joblib",
    "b1": "C:/Users/your_user/Documents/results/b1/joblib/model_b1.joblib",
    "b2": "C:/Users/your_user/Documents/results/b2/joblib/model_b2.joblib",
    "c3": "C:/Users/your_user/Documents/results/c3/joblib/model_c3.joblib",
}

# Load LightGBM models
print("Loading LightGBM models...")
loaded_lgb_models = {}
for model_name, path in lgb_models.items():
    if os.path.exists(path):
        try:
            loaded_lgb_models[model_name] = lgb.Booster(model_file=path)
            print(f"Loaded LightGBM model: {model_name}")
        except Exception as e:
            print(f"Failed to load LightGBM model {model_name}: {e}")
    else:
        print(f"Path not found for LightGBM model {model_name}: {path}")

# Load Joblib models
print("\nLoading Joblib models...")
loaded_joblib_models = {}
for model_name, path in joblib_models.items():
    if os.path.exists(path):
        try:
            loaded_joblib_models[model_name] = joblib.load(path)
            print(f"Loaded Joblib model: {model_name}")
        except Exception as e:
            print(f"Failed to load Joblib model {model_name}: {e}")
    else:
        print(f"Path not found for Joblib model {model_name}: {path}")

# Verify loaded models
print("\nVerifying loaded models...")
for model_name, model in loaded_lgb_models.items():
    print(f"LightGBM model {model_name}: Type = {type(model)}")

for model_name, model in loaded_joblib_models.items():
    print(f"Joblib model {model_name}: Type = {type(model)}")

# Create a DataFrame for predictions
predictions = pd.DataFrame()
predictions["index"] = df_test.index  # Add index for reference

# Scoring with LightGBM models
print("\nScoring with LightGBM models...")
for model_name, model in loaded_lgb_models.items():
    try:
        if isinstance(model, lgb.Booster):
            predictions[f"{model_name}_lgb_prediction"] = model.predict(df_test)
            print(f"Scored with LightGBM model: {model_name}")
        else:
            print(f"Invalid LightGBM model type for {model_name}: {type(model)}")
    except Exception as e:
        print(f"Failed to score with LightGBM model {model_name}: {e}")

# Scoring with Joblib models
print("\nScoring with Joblib models...")
for model_name, model in loaded_joblib_models.items():
    try:
        if hasattr(model, "predict"):
            predictions[f"{model_name}_joblib_prediction"] = model.predict(df_test)
            print(f"Scored with Joblib model: {model_name}")
        else:
            print(f"Invalid Joblib model type for {model_name}: {type(model)}")
    except Exception as e:
        print(f"Failed to score with Joblib model {model_name}: {e}")

# Save predictions to CSV
output_file = "model_predictions.csv"
try:
    predictions.to_csv(output_file, index=False)
    print(f"\nPredictions saved successfully to {output_file}")
except Exception as e:
    print(f"Failed to save predictions to CSV: {e}")

