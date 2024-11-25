from sklearn.model_selection import cross_val_score

# Cross-validation
cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())



from sklearn.model_selection import StratifiedKFold

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# Cross-validation
cv_scores_stratified = cross_val_score(model, X_processed, y, cv=skf, scoring='accuracy')
print("Stratified CV Accuracy Scores:", cv_scores_stratified)
print("Mean Stratified CV Accuracy:", cv_scores_stratified.mean())



#Use SMOTE for oversampling the minority class.
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample
