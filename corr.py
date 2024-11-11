def plot_correlation_matrix(self,
                              df: pd.DataFrame,
                              columns: Optional[List[str]] = None,
                              min_correlation: float = 0.0,
                              figsize: Tuple[int, int] = (20, 16),
                              save_pdf: Optional[str] = None,
                              cmap: str = 'coolwarm') -> Dict:
        """
        Create and save correlation matrix visualization with filtering options.
        
        Args:
            df: Input DataFrame
            columns: Specific columns to analyze (if None, uses all numeric)
            min_correlation: Minimum absolute correlation value to display
            figsize: Figure size in inches
            save_pdf: Filename for saving PDF
            cmap: Color map for correlation matrix
            
        Returns:
            Dictionary containing correlation statistics
        """
        numeric_cols = columns or df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in the dataset")
            
        print(f"\nCalculating correlations for {len(numeric_cols)} numeric columns...")
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Filter correlations based on minimum threshold
        mask = np.abs(corr_matrix) < min_correlation
        corr_matrix_filtered = corr_matrix.copy()
        corr_matrix_filtered[mask] = np.nan
        
        # Create PDF if requested
        pdf_path = self.output_dir / save_pdf if save_pdf else None
        pdf_handle = PdfPages(pdf_path) if pdf_path else None
        
        try:
            # For large feature sets, split into multiple plots
            max_features_per_plot = 50
            total_features = len(numeric_cols)
            num_plots = (total_features + max_features_per_plot - 1) // max_features_per_plot
            
            for plot_idx in range(num_plots):
                start_idx = plot_idx * max_features_per_plot
                end_idx = min(start_idx + max_features_per_plot, total_features)
                current_cols = numeric_cols[start_idx:end_idx]
                
                # Create correlation plot
                plt.figure(figsize=figsize)
                sns.heatmap(corr_matrix_filtered.loc[current_cols, current_cols],
                           cmap=cmap,
                           center=0,
                           annot=True,
                           fmt='.2f',
                           square=True,
                           cbar_kws={'label': 'Correlation Coefficient'})
                
                plt.title(f'Correlation Matrix (Part {plot_idx + 1} of {num_plots})\n'
                         f'Showing correlations with absolute value >= {min_correlation}')
                plt.tight_layout()
                
                if pdf_handle:
                    pdf_handle.savefig(plt.gcf(), dpi=self.dpi, bbox_inches='tight')
                plt.show()
                plt.close()
                
                print(f"Processed correlation matrix part {plot_idx + 1} of {num_plots}")
            
            # Save correlation matrix to CSV
            if save_pdf:
                corr_filename = save_pdf.replace('.pdf', '_correlation.csv')
                corr_path = self.logs_dir / f"{corr_filename}_{self.timestamp}.csv"
                corr_matrix.to_csv(corr_path)
                print(f"\nCorrelation matrix saved to: {corr_path}")
                
        finally:
            if pdf_handle:
                pdf_handle.close()
                if save_pdf:
                    print(f"\nPlots saved to: {pdf_path}")
        
        # Return correlation statistics
        correlation_stats = {
            'num_features': len(numeric_cols),
            'avg_correlation': float(np.nanmean(np.abs(corr_matrix_filtered))),
            'max_correlation': float(np.nanmax(np.abs(corr_matrix_filtered))),
            'min_correlation': float(np.nanmin(np.abs(corr_matrix_filtered[corr_matrix_filtered != 0]))),
            'num_strong_correlations': int(np.sum(np.abs(corr_matrix_filtered) > 0.7))
        }
        
        return correlation_stats
