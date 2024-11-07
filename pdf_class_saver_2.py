import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import gc
from typing import Optional, List, Union, Dict
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import sys
from tabulate import tabulate  # for nice terminal tables

class DataVisualizer:
    """
    Enhanced utility class for data visualization with terminal output 
    and file saving capabilities.
    """
    
    def __init__(self, 
                 style: str = 'seaborn',
                 output_dir: Optional[str] = None,
                 dpi: int = 300):
        """Initialize visualizer with output settings."""
        plt.style.use(style)
        self.dpi = dpi
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup timestamp for file naming
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _save_statistics(self, stats: Dict, filename: str) -> None:
        """Save statistics to JSON file."""
        file_path = self.logs_dir / f"{filename}_{self.timestamp}.json"
        with open(file_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"\nStatistics saved to: {file_path}")
    
    def _print_statistics(self, stats: Dict, title: str) -> None:
        """Print formatted statistics to terminal."""
        print(f"\n{'-'*20} {title} {'-'*20}")
        
        # Convert stats to table format
        table_data = []
        for col, col_stats in stats.items():
            row = [col]
            row.extend([f"{v:.2f}" if isinstance(v, (float, np.floating)) else v 
                       for v in col_stats.values()])
            table_data.append(row)
        
        # Get headers
        headers = ['Column'] + list(next(iter(stats.values())).keys())
        
        # Print table
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def plot_numeric_distributions(self, 
                                 df: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 plots_per_page: int = 6,
                                 figsize: tuple = (15, 10),
                                 save_pdf: Optional[str] = None,
                                 show_kde: bool = True) -> Dict[str, Dict]:
        """Enhanced distribution plotting with comprehensive output."""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist() \
                      if columns is None else \
                      [col for col in columns if df[col].dtype in ['int64', 'float64']]
        
        if not numeric_cols:
            raise ValueError("No numeric columns found in the dataset")
        
        print(f"\nAnalyzing {len(numeric_cols)} numeric columns...")
        
        # Initialize statistics dictionary
        stats_dict = {}
        
        # Setup PDF
        pdf_path = self.output_dir / save_pdf if save_pdf else None
        pdf_handle = PdfPages(pdf_path) if pdf_path else None
        
        try:
            total_figures = (len(numeric_cols) + plots_per_page - 1) // plots_per_page
            
            for fig_num in range(total_figures):
                start_idx = fig_num * plots_per_page
                batch_cols = numeric_cols[start_idx:start_idx + plots_per_page]
                
                # Create subplots
                fig, axes = plt.subplots(plots_per_page // 2, 2, figsize=figsize)
                axes = axes.ravel()
                
                # Process each column in batch
                for idx, col in enumerate(batch_cols):
                    ax = axes[idx]
                    data = df[col].dropna()
                    
                    # Plot distribution
                    sns.histplot(data=data, ax=ax, kde=show_kde)
                    
                    # Customize plot
                    ax.set_title(f'Distribution of {col}')
                    ax.grid(True, alpha=0.3)
                    
                    # Calculate statistics
                    stats = {
                        'mean': data.mean(),
                        'median': data.median(),
                        'std': data.std(),
                        'skew': data.skew(),
                        'kurtosis': data.kurtosis(),
                        'missing': df[col].isnull().sum(),
                        'missing_pct': (df[col].isnull().sum() / len(df)) * 100,
                        'unique_values': df[col].nunique(),
                        'min': data.min(),
                        'max': data.max()
                    }
                    
                    # Add stats to plot
                    stats_text = "\n".join([f"{k}: {v:.2f}" if isinstance(v, (float, np.floating))
                                          else f"{k}: {v}" 
                                          for k, v in stats.items()])
                    ax.text(0.95, 0.95, stats_text,
                           transform=ax.transAxes,
                           verticalalignment='top',
                           horizontalalignment='right',
                           bbox=dict(facecolor='white', alpha=0.8))
                    
                    stats_dict[col] = stats
                
                # Handle empty subplots
                for idx in range(len(batch_cols), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                
                if pdf_handle:
                    pdf_handle.savefig(fig, dpi=self.dpi)
                
                plt.show()
                plt.close()
                gc.collect()
                
                print(f"Processed figure {fig_num + 1} of {total_figures}")
            
            # Print and save statistics
            self._print_statistics(stats_dict, "Numeric Columns Statistics")
            if save_pdf:
                stats_filename = save_pdf.replace('.pdf', '_stats')
                self._save_statistics(stats_dict, stats_filename)
            
        except Exception as e:
            print(f"Error during plotting: {str(e)}", file=sys.stderr)
            raise
        finally:
            if pdf_handle:
                pdf_handle.close()
                print(f"\nPlots saved to: {pdf_path}")
        
        return stats_dict
    
    def plot_correlation_heatmap(self,
                               df: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               figsize: tuple = (12, 10),
                               save_pdf: Optional[str] = None,
                               min_correlation: float = 0.0) -> pd.DataFrame:
        """Enhanced correlation heatmap with detailed output."""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist() \
                      if columns is None else \
                      [col for col in columns if df[col].dtype in ['int64', 'float64']]
        
        print(f"\nCalculating correlations for {len(numeric_cols)} columns...")
        
        # Calculate correlations
        corr_matrix = df[numeric_cols].corr()
        
        # Filter correlations if requested
        if min_correlation > 0:
            mask = np.abs(corr_matrix) >= min_correlation
            corr_matrix = corr_matrix.where(mask, 0)
        
        # Plot heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix,
                   annot=True,
                   cmap='coolwarm',
                   center=0,
                   fmt='.2f',
                   square=True)
        
        plt.title('Feature Correlation Heatmap')
        
        # Save if requested
        if save_pdf:
            pdf_path = self.output_dir / save_pdf
            plt.savefig(pdf_path, bbox_inches='tight', dpi=self.dpi)
            print(f"\nCorrelation heatmap saved to: {pdf_path}")
            
            # Save correlation matrix
            matrix_filename = save_pdf.replace('.pdf', '_matrix')
            corr_dict = corr_matrix.to_dict()
            self._save_statistics(corr_dict, matrix_filename)
        
        # Find and print high correlations
        high_corr = corr_matrix[abs(corr_matrix) >= 0.7].unstack()
        high_corr = high_corr[high_corr != 1.0].dropna()
        
        if not high_corr.empty:
            print("\nHigh Correlations (|r| >= 0.7):")
            high_corr_data = [(idx[0], idx[1], val) for idx, val in high_corr.items()]
            print(tabulate(high_corr_data, 
                         headers=['Feature 1', 'Feature 2', 'Correlation'],
                         tablefmt='grid',
                         floatfmt='.2f'))
        
        plt.show()
        plt.close()
        
        return corr_matrix

# Example usage
if __name__ == "__main__":
    # Create sample data
    df = pd.DataFrame({
        'A': np.random.normal(0, 1, 1000),
        'B': np.random.exponential(2, 1000),
        'C': np.random.uniform(0, 10, 1000),
        'D': np.random.chisquare(5, 1000)
    })
    
    # Initialize visualizer
    viz = DataVisualizer(output_dir='visualization_output')
    
    # Plot distributions
    stats = viz.plot_numeric_distributions(
        df,
        save_pdf='distributions.pdf'
    )
    
    # Plot correlations
    corr_matrix = viz.plot_correlation_heatmap(
        df,
        save_pdf='correlations.pdf',
        min_correlation=0.3
    )
