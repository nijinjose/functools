import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_vehicle_maa_distribution(df: pd.DataFrame, col_advfe: str, col_target: str):
    """
    Analyzes the distribution of the 'vehicle_maa' feature and provides insights
    to help select appropriate parameters for over-indexing and under-indexing.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features and target.
    col_advfe : str
        The name of the categorical feature to analyze (e.g., 'vehicle_maa').
    col_target : str
        The name of the target variable (e.g., 'churn_flag').
    
    Returns
    -------
    dict
        A dictionary containing suggested parameter values and relevant statistics.
    """
    # Fill missing values
    df[col_advfe] = df[col_advfe].fillna('Unknown')
    
    # Compute basic statistics
    total_obs = df.shape[0]
    unique_categories = df[col_advfe].nunique()
    category_counts = df[col_advfe].value_counts()
    
    print(f"Total observations: {total_obs}")
    print(f"Unique categories in '{col_advfe}': {unique_categories}")
    
    # Plot the distribution of category counts
    plt.figure(figsize=(12, 6))
    sns.histplot(category_counts.values, bins=50, log_scale=(True, True))
    plt.title(f"Distribution of '{col_advfe}' Category Counts")
    plt.xlabel("Number of Observations per Category")
    plt.ylabel("Number of Categories")
    plt.show()
    
    # Analyze the cumulative distribution
    cumulative_counts = category_counts.sort_values(ascending=False).cumsum()
    cumulative_percentage = 100 * cumulative_counts / total_obs
    
    # Plot cumulative percentage of observations covered by top N categories
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(cumulative_percentage)+1), cumulative_percentage.values)
    plt.title(f"Cumulative Coverage of Top N Categories in '{col_advfe}'")
    plt.xlabel("Number of Top Categories")
    plt.ylabel("Cumulative Percentage of Total Observations")
    plt.grid(True)
    plt.show()
    
    # Suggest min_obs_over and min_obs_under based on distribution
    # For example, select categories covering at least 80% of observations
    coverage_threshold = 80  # Percentage
    categories_to_cover = cumulative_percentage[cumulative_percentage <= coverage_threshold].index.tolist()
    min_obs_suggested = category_counts[categories_to_cover[-1]]
    
    print(f"Suggested 'min_obs_over' and 'min_obs_under': {min_obs_suggested}")
    
    # Suggest n_fea_over and n_fea_under based on churn rates
    # Compute churn rates per category
    category_churn = df.groupby(col_advfe)[col_target].mean().sort_values(ascending=False)
    
    # Identify categories with highest and lowest churn rates
    n_fea_suggested = 20  # Default value, can be adjusted
    
    # Ensure n_fea_suggested does not exceed the number of available categories
    n_fea_suggested = min(n_fea_suggested, unique_categories)
    
    print(f"Suggested 'n_fea_over' and 'n_fea_under': {n_fea_suggested}")
    
    # Return suggestions and statistics
    suggestions = {
        'min_obs_over': min_obs_suggested,
        'min_obs_under': min_obs_suggested,
        'n_fea_over': n_fea_suggested,
        'n_fea_under': n_fea_suggested,
        'total_observations': total_obs,
        'unique_categories': unique_categories,
        'category_counts': category_counts,
        'category_churn_rates': category_churn
    }
    
    return suggestions

# Usage example:
# Assuming 'df' is your DataFrame, 'vehicle_maa' is your feature, and 'churn_flag' is your target
suggestions = analyze_vehicle_maa_distribution(vdf, 'Vehicle', 'Misc_Cancelled')

# You can now use 'suggestions' to set parameters in your over_under_index function
min_obs_over = suggestions['min_obs_over']
min_obs_under = suggestions['min_obs_under']
n_fea_over = suggestions['n_fea_over']
n_fea_under = suggestions['n_fea_under']
