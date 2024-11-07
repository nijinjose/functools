import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import seaborn as sns
import warnings
from typing import Optional, List, Dict, Tuple

# Suppress warnings
warnings.filterwarnings("ignore")

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

class DataVisualizer:
    """
    A comprehensive class for visualizing and analyzing numerical data from DataFrames.
    Creates distribution plots and saves detailed statistics.
    """
    
    def __init__(self, 
                 output_dir: Optional[str] = None, 
                 dpi: int = 300,
                 style: str = 'seaborn'):
        """
        Initialize the DataVisualizer.
        
        Args:
            output_dir: Directory to save outputs
            dpi: DPI for saved plots
            style: Plot style ('seaborn', 'default', etc.)
        """
        self.dpi = dpi
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.style.use(style)
    
    def _calculate_statistics(self, data: pd.Series) -> dict:
        """Calculate comprehensive statistics for a series."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Basic statistics
            stats = data.describe().to_dict()
            
            # Additional statistics
            stats.update({
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis()),
                'missing': int(data.isnull().sum()),
                'missing_pct': float(data.isnull().mean() * 100),
                'unique': int(data.nunique())
            })
            
            # Convert all values to appropriate types
            return {k: float(v) if isinstance(v, (np.floating, float)) 
                   else int(v) if isinstance(v, (np.integer, int)) 
                   else v 
                   for k, v in stats.items()}
    
    def _print_statistics(self, stats: dict, title: str = "Numeric Columns Statistics") -> None:
        """Print complete statistics to terminal."""
        print(f"\n{'-'*20} {title} {'-'*20}")
        
        # Create DataFrame with all statistics
        stats_df = pd.DataFrame(stats).transpose()
        
        # Order columns for better readability
        display_columns = [
            'count', 'unique', 'missing', 'missing_pct',
            'mean', 'std', 'min', '25%', '50%', '75%', 'max',
            'skewness', 'kurtosis'
        ]
        
        # Select and format columns
        stats_df = stats_df[display_columns]
        
        # Print with formatted floating points
        with pd.option_context('display.float_format', '{:.2f}'.format):
            print("\nComplete Statistics:")
            print(stats_df)
    
    def _save_statistics(self, stats: dict, filename: str, format: str = 'json') -> None:
        """Save statistics to file."""
        file_path = self.logs_dir / f"{filename}_{self.timestamp}.{format}"
        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump(stats, f, cls=NumpyEncoder, indent=4)
        elif format == 'csv':
            stats_df = pd.DataFrame(stats).transpose()
            stats_df.to_csv(file_path)
        print(f"\nStatistics saved to: {file_path}")

    def plot_numeric_distributions(self, 
                                 df: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 plots_per_page: int = 6,
                                 figsize: Tuple[int, int] = (15, 10),
                                 save_pdf: Optional[str] = None,
                                 show_kde: bool = True,
                                 color: str = 'skyblue',
                                 stats_format: str = 'json') -> Dict:
        """
        Plot distributions of numeric columns with complete statistics.
        
        Args:
            df: Input DataFrame
            columns: Specific columns to plot (if None, uses all numeric)
            plots_per_page: Number of plots per page
            figsize: Figure size in inches
            save_pdf: Filename for saving PDF
            show_kde: Whether to show KDE curve
            color: Color for histograms
            stats_format: Format for saving statistics ('json' or 'csv')
            
        Returns:
            Dictionary containing statistics for each column
        """
        numeric_cols = columns or df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in the dataset")
            
        print(f"\nAnalyzing {len(numeric_cols)} numeric columns...")
        
        stats_dict = {}
        pdf_path = self.output_dir / save_pdf if save_pdf else None
        pdf_handle = PdfPages(pdf_path) if pdf_path else None
        
        try:
            total_figures = (len(numeric_cols) + plots_per_page - 1) // plots_per_page
            
            for fig_num in range(total_figures):
                start_idx = fig_num * plots_per_page
                batch_cols = numeric_cols[start_idx:start_idx + plots_per_page]
                
                fig, axes = plt.subplots(plots_per_page // 2, 2, figsize=figsize)
                axes = axes.ravel()
                
                for idx, col in enumerate(batch_cols):
                    ax = axes[idx]
                    data = df[col]
                    
                    # Calculate statistics
                    stats = self._calculate_statistics(data)
                    stats_dict[col] = stats
                    
                    # Create histogram with optional KDE
                    sns.histplot(data, ax=ax, kde=show_kde, bins=30,
                               color=color, edgecolor='black')
                    
                    ax.set_title(f'Distribution of {col}')
                    ax.axvline(stats['mean'], color='red', linestyle='--',
                             label=f"Mean: {stats['mean']:.2f}")
                    ax.axvline(stats['50%'], color='green', linestyle='--',
                             label=f"Median: {stats['50%']:.2f}")
                    ax.legend(fontsize='small')
                    
                    # Comprehensive statistics text
                    stats_text = (
                        f"Observations:\n"
                        f"Total: {stats['count']:.0f}\n"
                        f"Missing: {stats['missing']:.0f} ({stats['missing_pct']:.1f}%)\n"
                        f"Unique: {stats['unique']:.0f}\n"
                        f"\nCentral Tendency:\n"
                        f"Mean: {stats['mean']:.2f}\n"
                        f"Median: {stats['50%']:.2f}\n"
                        f"\nDispersion:\n"
                        f"Std Dev: {stats['std']:.2f}\n"
                        f"Min: {stats['min']:.2f}\n"
                        f"25%: {stats['25%']:.2f}\n"
                        f"75%: {stats['75%']:.2f}\n"
                        f"Max: {stats['max']:.2f}\n"
                        f"\nShape:\n"
                        f"Skewness: {stats['skewness']:.2f}\n"
                        f"Kurtosis: {stats['kurtosis']:.2f}"
                    )
                    
                    # Add statistics to plot
                    ax.text(1.02, 0.98, stats_text,
                           transform=ax.transAxes,
                           verticalalignment='top',
                           horizontalalignment='left',
                           bbox=dict(facecolor='white', alpha=0.8),
                           fontsize='x-small')
                
                # Hide empty subplots
                for idx in range(len(batch_cols), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                
                if pdf_handle:
                    pdf_handle.savefig(fig, dpi=self.dpi, bbox_inches='tight')
                plt.show()
                plt.close()
                
                print(f"Processed figure {fig_num + 1} of {total_figures}")
            
            # Print and save statistics
            self._print_statistics(stats_dict)
            if save_pdf:
                stats_filename = save_pdf.replace('.pdf', '_stats')
                self._save_statistics(stats_dict, stats_filename, format=stats_format)
                
        finally:
            if pdf_handle:
                pdf_handle.close()
                if save_pdf:
                    print(f"\nPlots saved to: {pdf_path}")
        
        return stats_dict

# Example usage
if __name__ == "__main__":
    # Create sample data with different distributions
    np.random.seed(42)
    df = pd.DataFrame({
        'Normal': np.random.normal(0, 1, 1000),
        'Skewed': np.random.exponential(2, 1000),
        'Bimodal': np.concatenate([
            np.random.normal(-2, 0.5, 500),
            np.random.normal(2, 0.5, 500)
        ]),
        'Uniform': np.random.uniform(-3, 3, 1000)
    })
    
    # Initialize visualizer
    viz = DataVisualizer(output_dir='visualization_output')
    
    # Create plots with all features
    stats = viz.plot_numeric_distributions(
        df,
        plots_per_page=4,
        show_kde=True,
        color='skyblue',
        save_pdf='distributions.pdf',
        stats_format='csv'
    )
