import os
import joblib
import lightgbm as lgb

# Define the base directory containing top-level folders
base_path = "C:/Users/your_user/Documents/results"  # Update this to your results folder

# Dictionaries to store models
lgb_models = {}
joblib_models = {}

# Traverse each top-level folder (a0, b1, b2, c3)
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    
    if os.path.isdir(folder_path):
        # Load the .lgb model from the main folder
        for file in os.listdir(folder_path):
            if file.endswith(".lgb"):
                lgb_model_path = os.path.join(folder_path, file)
                print(f"Loading LightGBM model from: {lgb_model_path}")
                try:
                    lgb_models[file] = lgb.Booster(model_file=lgb_model_path)
                    print(f"Loaded LightGBM model: {file}")
                except Exception as e:
                    print(f"Failed to load LightGBM model {file}: {e}")
        
        # Load the .joblib models from the 'joblib' subfolder
        joblib_folder_path = os.path.join(folder_path, "joblib")
        if os.path.exists(joblib_folder_path) and os.path.isdir(joblib_folder_path):
            for file in os.listdir(joblib_folder_path):
                if file.endswith(".joblib"):
                    joblib_model_path = os.path.join(joblib_folder_path, file)
                    print(f"Loading joblib model from: {joblib_model_path}")
                    try:
                        joblib_models[file] = joblib.load(joblib_model_path)
                        print(f"Loaded joblib model: {file}")
                    except Exception as e:
                        print(f"Failed to load joblib model {file}: {e}")

# Display all loaded models
print("Loaded LightGBM models:", list(lgb_models.keys()))
print("Loaded joblib models:", list(joblib_models.keys()))
