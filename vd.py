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



def analyze_van_churn_v3(df):
    try:
        # Step 1: Create empty DataFrame
        van_analysis = pd.DataFrame()
        
        # Step 2: Add each metric one by one
        van_analysis['vehicle_maa'] = df['vehicle_maa'].unique()
        
        # Step 3: Add counts
        count_series = df['vehicle_maa'].value_counts()
        van_analysis['total_vans'] = van_analysis['vehicle_maa'].map(count_series)
        
        # Step 4: Add churn rates
        churn_means = df.groupby('vehicle_maa')['churn'].mean()
        van_analysis['churn_rate'] = van_analysis['vehicle_maa'].map(churn_means).round(4)
        
        # Step 5: Sort and display
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
        print(f"Error: {str(e)}")
        return None






def analyze_van_churn_v3(df):
    try:
        # Step 1: Create empty DataFrame
        van_analysis = pd.DataFrame()
        
        # Step 2: Add each metric one by one
        van_analysis['vehicle_maa'] = df['vehicle_maa'].unique()
        
        # Step 3: Add counts
        count_series = df['vehicle_maa'].value_counts()
        van_analysis['total_vans'] = van_analysis['vehicle_maa'].map(count_series)
        
        # Step 4: Add churn rates
        churn_means = df.groupby('vehicle_maa')['churn'].mean()
        van_analysis['churn_rate'] = van_analysis['vehicle_maa'].map(churn_means).round(4)
        
        # Step 5: Sort and display
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
        print(f"Error: {str(e)}")
        return None






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
