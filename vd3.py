def analyze_van_churn(df):
    try:
        print("Starting analysis...")
        
        # Create results dictionary step by step
        results = []
        
        # Get unique vehicles and their stats
        for vehicle in df['vehicle_maa'].unique():
            vehicle_data = df[df['vehicle_maa'] == vehicle]
            results.append({
                'vehicle_maa': vehicle,
                'total_vans': len(vehicle_data),
                'churn_rate': float(vehicle_data['churn'].mean().round(4))  # Convert to float to avoid Series
            })
        
        # Convert to DataFrame
        van_analysis = pd.DataFrame(results)
        
        # Sort for high churn - using nlargest/nsmallest instead of sort_values
        print("\nChurn Analysis:")
        print("-" * 50)
        
        print("\nTop 5 High Churn Vans:")
        high_churn = van_analysis.nlargest(5, 'churn_rate')
        print(high_churn[['vehicle_maa', 'total_vans', 'churn_rate']])
        
        print("\nTop 5 Low Churn Vans:")
        low_churn = van_analysis.nsmallest(5, 'churn_rate')
        print(low_churn[['vehicle_maa', 'total_vans', 'churn_rate']])
        
        return van_analysis
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

# Alternative simpler version using direct groupby
def analyze_van_churn_v2(df):
    try:
        # Get stats using groupby
        stats = df.groupby('vehicle_maa').agg({
            'vehicle_maa': 'size',    # Count of vans
            'churn': 'mean'           # Churn rate
        }).reset_index()
        
        # Rename columns
        stats.columns = ['vehicle_maa', 'total_vans', 'churn_rate']
        
        # Round churn rate
        stats['churn_rate'] = stats['churn_rate'].round(4)
        
        # Display results
        print("\nChurn Analysis:")
        print("-" * 50)
        
        print("\nTop 5 High Churn Vans:")
        print(stats.nlargest(5, 'churn_rate')[['vehicle_maa', 'total_vans', 'churn_rate']])
        
        print("\nTop 5 Low Churn Vans:")
        print(stats.nsmallest(5, 'churn_rate')[['vehicle_maa', 'total_vans', 'churn_rate']])
        
        return stats
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

# Let's try both versions with your data
print("Trying first version:")
result1 = analyze_van_churn(df)

print("\nTrying second version:")
result2 = analyze_van_churn_v2(df)
