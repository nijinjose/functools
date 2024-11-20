def analyze_van_churn_debug(df):
    # First, let's print what we're working with
    print("Initial DataFrame Structure:")
    print("-" * 50)
    print("Columns:", df.columns.tolist())
    print("\nSample data first 5 rows:")
    print(df.head())
    
    try:
        # Method 1: Direct calculation
        print("\nMethod 1 Results:")
        print("-" * 50)
        
        # Get unique vehicles
        unique_vehicles = df['vehicle_maa'].unique()
        print(f"Number of unique vehicles: {len(unique_vehicles)}")
        
        # Create results dictionary
        results = []
        
        for vehicle in unique_vehicles:
            vehicle_data = df[df['vehicle_maa'] == vehicle]
            results.append({
                'vehicle_maa': vehicle,
                'total_vans': len(vehicle_data),
                'churn_rate': vehicle_data['churn'].mean().round(4)
            })
        
        # Convert to DataFrame
        van_analysis = pd.DataFrame(results)
        
        # Sort and display
        print("\nFinal Results:")
        print("-" * 50)
        print("\nTop 5 High Churn Vans:")
        print(van_analysis.sort_values('churn_rate', ascending=False).head())
        print("\nTop 5 Low Churn Vans:")
        print(van_analysis.sort_values('churn_rate', ascending=True).head())
        
        return van_analysis
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nDetailed error information:")
        import traceback
        print(traceback.format_exc())
        return None

# Let's try another version if the above doesn't work
def analyze_van_churn_simple(df):
    try:
        # Step 1: Just get the counts
        print("Step 1: Getting counts...")
        counts = df['vehicle_maa'].value_counts()
        print(f"Found {len(counts)} unique vehicles")
        
        # Step 2: Just get the mean churn
        print("\nStep 2: Calculating churn rates...")
        churn = df.groupby('vehicle_maa')['churn'].mean()
        print(f"Calculated churn rates for {len(churn)} vehicles")
        
        # Step 3: Combine into new dataframe
        print("\nStep 3: Creating final dataframe...")
        final_df = pd.DataFrame({
            'total_vans': counts,
            'churn_rate': churn.round(4)
        }).reset_index()
        
        final_df.columns = ['vehicle_maa', 'total_vans', 'churn_rate']
        
        # Step 4: Sort and display
        print("\nResults:")
        print("-" * 50)
        print("\nTop 5 High Churn Vans:")
        print(final_df.nlargest(5, 'churn_rate')[['vehicle_maa', 'total_vans', 'churn_rate']])
        print("\nTop 5 Low Churn Vans:")
        print(final_df.nsmallest(5, 'churn_rate')[['vehicle_maa', 'total_vans', 'churn_rate']])
        
        return final_df
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nDetailed error information:")
        import traceback
        print(traceback.format_exc())
        return None

# Try both versions and see which works:
print("Trying debug version:")
result1 = analyze_van_churn_debug(df)

print("\nTrying simple version:")
result2 = analyze_van_churn_simple(df)

# If both fail, let's at least see what we're working with:
print("\nDataFrame Information:")
print("-" * 50)
print("\nDataFrame Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nColumn Types:")
print(df.dtypes)
