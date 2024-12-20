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

# Display loaded models
print("Loaded LightGBM models:", list(loaded_lgb_models.keys()))
print("Loaded joblib models:", list(loaded_joblib_models.keys()))
