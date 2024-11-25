from pandas_profiling import ProfileReport

profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_notebook_iframe()







from sklearn.feature_selection import SelectKBest, f_classif

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Apply SelectKBest
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features)






#(RFE) with a classifier.
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Initialize model
model = RandomForestClassifier()

# Initialize RFE
rfe = RFE(estimator=model, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

# Get selected feature names
selected_features_rfe = X.columns[rfe.support_]
print("Selected Features via RFE:", selected_features_rfe)
