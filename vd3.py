group_means = vdf.groupby('Vehicle')['Misc_Cancelled'].mean()
print(group_means.head())
print(type(group_means))


sorted_means = group_means.sort_values(ascending=False)
print(sorted_means.head())
print(type(sorted_means))


# Compute churn rates per category
try:
    category_churn = df.groupby(col_advfe)[col_target].mean()
    print(f"Groupby result (before sorting):\n{category_churn.head()}\n")

    # Sort the churn rates
    category_churn = category_churn.sort_values(ascending=False)
    print(f"Sorted churn rates:\n{category_churn.head()}\n")
except Exception as e:
    print(f"Error encountered while calculating or sorting churn rates: {e}")


print(vdf['Misc_Cancelled'].dtype)


vdf['Misc_Cancelled'] = pd.to_numeric(vdf['Misc_Cancelled'], errors='coerce')


vdf['Misc_Cancelled'] = vdf['Misc_Cancelled'].fillna(0)  # Or another appropriate value


print(vdf['Vehicle'].isna().sum())  # Check for missing values
print(vdf['Vehicle'].unique())     # Inspect unique values



# Compute churn rates per category
try:
    # Ensure the target column is numeric
    df[col_target] = pd.to_numeric(df[col_target], errors='coerce')
    
    # Fill missing values in the target column
    df[col_target] = df[col_target].fillna(0)
    
    # Calculate churn rates
    category_churn = df.groupby(col_advfe)[col_target].mean()
    print(f"Groupby result (before sorting):\n{category_churn.head()}\n")

    # Sort churn rates
    category_churn = category_churn.sort_values(ascending=False)
    print(f"Sorted churn rates:\n{category_churn.head()}\n")
except Exception as e:
    print(f"Error encountered while calculating or sorting churn rates: {e}")
