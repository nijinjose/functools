import pandas as pd

# Assume df1 and df2 are already defined
# df1: Original DataFrame
# df2: Modified DataFrame

# Step 1: Compare column differences
columns_removed = set(df1.columns) - set(df2.columns)
columns_added = set(df2.columns) - set(df1.columns)

print(f"Columns removed ({len(columns_removed)}): {columns_removed}")
print(f"Columns added ({len(columns_added)}): {columns_added}")

# Step 2: Compare data differences in common columns
common_columns = set(df1.columns).intersection(set(df2.columns))
data_differences = {}
difference_counts = {}

for col in common_columns:
    if pd.api.types.is_categorical_dtype(df1[col]) and pd.api.types.is_categorical_dtype(df2[col]):
        # Ensure categories are the same before comparing
        if set(df1[col].cat.categories) != set(df2[col].cat.categories):
            print(f"Column '{col}' has different categories and cannot be directly compared.")
        else:
            # Compare only values
            diff_mask = df1[col] != df2[col]
            count_diff = diff_mask.sum()
            if count_diff > 0:
                data_differences[col] = df1.loc[diff_mask, col].reset_index().join(
                    df2.loc[diff_mask, col].reset_index(), lsuffix='_df1', rsuffix='_df2'
                )
                difference_counts[col] = count_diff
    else:
        # Handle non-categorical columns
        if not df1[col].equals(df2[col]):
            diff_mask = df1[col] != df2[col]
            count_diff = diff_mask.sum()
            if count_diff > 0:
                data_differences[col] = df1.loc[diff_mask, col].reset_index().join(
                    df2.loc[diff_mask, col].reset_index(), lsuffix='_df1', rsuffix='_df2'
                )
                difference_counts[col] = count_diff

# Step 3: Report data differences summary
if data_differences:
    print(f"\nData differences found in {len(data_differences)} common columns:")
    for col, count in difference_counts.items():
        print(f"- Column '{col}': {count} rows with differences")
else:
    print("\nNo data differences in common columns.")

# Optional: Save detailed differences to files for review
for col, diff in data_differences.items():
    diff.to_csv(f"differences_in_{col}.csv", index=False)
    print(f"Differences in column '{col}' saved to 'differences_in_{col}.csv'")
