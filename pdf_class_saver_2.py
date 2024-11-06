import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import gc
from typing import Optional, List, Union
import seaborn as sns

class DataVisualizer:
    """
    Utility class for data visualization with support for both 
    notebook display and PDF saving.
    """
    
    def __init__(self, style: str = 'seaborn'):
        """
        Initialize visualizer with matplotlib style.
        
        Args:
            style: matplotlib style to use ('seaborn', 'fivethirtyeight', etc.)
        """
        plt.style.use(style)
        
    def plot_numeric_distributions(self, 
                                 df: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 plots_per_page: int = 6,
                                 figsize: tuple = (15, 10),
                                 save_pdf: Optional[str] = None) -> None:
        """
        Plot distributions of numeric columns. Can both display in notebook
        and save to PDF.
        
        Args:
            df: DataFrame containing the data
            columns: Specific columns to plot (if None, uses all numeric)
            plots_per_page: Number of plots to show per figure
            figsize: Size of the figure
            save_pdf: If provided, saves plots to this PDF file
        """
        # Get numeric columns if not specified
        if columns is None:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        else:
            numeric_cols = [col for col in columns if df[col].dtype in ['int64', 'float64']]
            
        print(f"Processing {len(numeric_cols)} numeric columns")
        
        # Calculate number of figures needed
        n_cols = 2
        n_rows = plots_per_page // 2
        total_figures = (len(numeric_cols) + plots_per_page - 1) // plots_per_page
        
        # Open PDF if needed
        pdf_handle = PdfPages(save_pdf) if save_pdf else None
        
        try:
            # Process columns in batches
            for fig_num in range(total_figures):
                start_idx = fig_num * plots_per_page
                batch_cols = numeric_cols[start_idx:start_idx + plots_per_page]
                
                # Create figure
                fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
                axes = axes.ravel()
                
                # Plot each column in batch
                for idx, col in enumerate(batch_cols):
                    ax = axes[idx]
                    
                    # Get column data
                    data = df[col].dropna()
                    
                    # Create histogram with KDE
                    sns.histplot(data=data, ax=ax, kde=True)
                    
                    # Add title and grid
                    ax.set_title(f'Distribution of {col}')
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics
                    stats = data.describe()
                    stats_text = (
                        f'Mean: {stats["mean"]:.2f}\n'
                        f'Std: {stats["std"]:.2f}\n'
                        f'Skew: {data.skew():.2f}\n'
                        f'Kurt: {data.kurtosis():.2f}'
                    )
                    ax.text(0.95, 0.95, stats_text,
                           transform=ax.transAxes,
                           verticalalignment='top',
                           horizontalalignment='right',
                           bbox=dict(facecolor='white', alpha=0.8))
                
                # Hide empty subplots
                for idx in range(len(batch_cols), len(axes)):
                    axes[idx].set_visible(False)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save to PDF if requested
                if pdf_handle:
                    pdf_handle.savefig(fig)
                
                # Show in notebook
                plt.show()
                plt.close()
                
                # Clean up
                gc.collect()
                
                print(f"Processed figure {fig_num + 1} of {total_figures}")
            
            if pdf_handle:
                print(f"Successfully saved plots to {save_pdf}")
                
        finally:
            # Ensure PDF is closed
            if pdf_handle:
                pdf_handle.close()
    
    def plot_correlation_heatmap(self,
                               df: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               figsize: tuple = (12, 10),
                               save_pdf: Optional[str] = None) -> None:
        """
        Plot correlation heatmap for numeric columns.
        
        Args:
            df: DataFrame containing the data
            columns: Specific columns to include (if None, uses all numeric)
            figsize: Size of the figure
            save_pdf: If provided, saves plot to this PDF file
        """
        # Get numeric columns if not specified
        if columns is None:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        else:
            numeric_cols = [col for col in columns if df[col].dtype in ['int64', 'float64']]
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   square=True)
        
        plt.title('Feature Correlation Heatmap')
        
        # Save if requested
        if save_pdf:
            plt.savefig(save_pdf, bbox_inches='tight')
        
        # Show in notebook
        plt.show()
        plt.close()

# Example usage in notebook:
if __name__ == '__main__':


# Usuage 
# 1. Just display in notebook (no PDF)
viz.plot_numeric_distributions(df, plots_per_page=6)

# 2. Display AND save to PDF
viz.plot_numeric_distributions(
    df,
    plots_per_page=6,
    save_pdf='distributions.pdf'
)

# 3. Plot specific columns only
selected_columns = ['feature1', 'feature2', 'feature3']
viz.plot_numeric_distributions(
    df,
    columns=selected_columns,
    plots_per_page=6
)

# 4. Plot correlation heatmap
viz.plot_correlation_heatmap(df, save_pdf='correlations.pdf')

# 5. Analyze subset of data
subset_df = df[df['some_column'] > some_value]
viz.plot_numeric_distributions(
    subset_df,
    save_pdf='subset_analysis.pdf'
)
