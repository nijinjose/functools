# Assuming you have a DataFrame with columns: vehicle_maa (van type) and churn (0 or 1)
import pandas as pd
import numpy as np

# Example check your data
def check_van_data(df):
    print("Data Overview:")
    print("-" * 50)
    print(f"Total records: {len(df)}")
    print(f"Total unique vans: {df['vehicle_maa'].nunique()}")
    
    # Show sample of your data
    print("\nFirst few records:")
    print(df[['vehicle_maa', 'churn']].head())

# Usage:
check_van_data(df)




def analyze_van_counts(df):
    # Count each van type
    van_counts = df['vehicle_maa'].value_counts()
    
    print("Van Type Distribution:")
    print("-" * 50)
    print("\nTop 10 Most Common Vans:")
    print(van_counts.head(10))
    
    return van_counts

# Usage:
van_counts = analyze_van_counts(df)


def analyze_van_churn(df):
    try:
        # Group by van type and calculate statistics
        van_analysis = df.groupby('vehicle_maa')['churn'].agg(['count', 'mean']).round(4)
        
        # Now columns will be ['count', 'mean']
        # Rename them properly
        van_analysis.columns = ['total_vans', 'churn_rate']
        
        # Sort by churn rate for both high and low churn
        high_churn = van_analysis.sort_values('churn_rate', ascending=False)
        low_churn = van_analysis.sort_values('churn_rate', ascending=True)
        
        print("Churn Analysis:")
        print("-" * 50)
        print("\nTop 5 High Churn Vans:")
        print(high_churn.head())
        print("\nTop 5 Low Churn Vans:")
        print(low_churn.head())
        
        return van_analysis
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        print("Debug information:")
        print(f"Columns after aggregation: {van_analysis.columns}")
        return None



# Run analysis
van_churn = analyze_van_churn(df_sample)



validation : 

def validate_results(df, selected_vans):
    # Calculate coverage
    total_vans = len(df)
    covered_vans = df[df['vehicle_maa'].isin(selected_vans)].shape[0]
    coverage = (covered_vans / total_vans) * 100
    
    print("Validation Results:")
    print("-" * 50)
    print(f"Selected van types: {len(selected_vans)}")
    print(f"Data coverage: {coverage:.1f}%")
    
    # Check distribution of remaining vans
    uncovered = df[~df['vehicle_maa'].isin(selected_vans)]
    if len(uncovered) > 0:
        print("\nTop 5 uncovered van types:")
        print(uncovered['vehicle_maa'].value_counts().head())

# Usage:
validate_results(df, results['high_churn'] + results['low_churn'])
