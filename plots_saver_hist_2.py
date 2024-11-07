import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import seaborn as sns
import warnings

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
    def __init__(self, output_dir=None, dpi=300):
        self.dpi = dpi
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set style for better visualization
        plt.style.use('seaborn')
        
    def _calculate_statistics(self, data: pd.Series) -> dict:
        """Calculate comprehensive statistics for a series."""
        stats = data.describe().to_dict()
        stats['skewness'] = data.skew()
        stats['kurtosis'] = data.kurtosis()
        stats['missing'] = data.isnull().sum()
        stats['missing_pct'] = data.isnull().mean() * 100
        stats['unique'] = data.nunique()
        return {k: float(v) for k, v in stats.items()}
    
    def _print_statistics(self, stats: dict, title: str = "Numeric Columns Statistics") -> None:
        """Print formatted statistics to the terminal."""
        print(f"\n{'-'*20} {title} {'-'*20}")
        if not stats:
            print("No statistics available.")
            return
        
        # Create a DataFrame from the stats dictionary
        stats_df = pd.DataFrame(stats).transpose()
        # Select and order the columns to display
        display_columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
                           'skewness', 'kurtosis', 'missing', 'missing_pct', 'unique']
        stats_df = stats_df[display_columns]
        print(stats_df)
    
    def _save_statistics(self, stats: dict, filename: str) -> None:
        """Save statistics to JSON file."""
        file_path = self.logs_dir / f"{filename}_{self.timestamp}.json"
        with open(file_path, 'w') as f:
            json.dump(stats, f, cls=NumpyEncoder, indent=4)
        print(f"\nStatistics saved to: {file_path}")

    def plot_numeric_distributions(self, 
                                   df: pd.DataFrame,
                                   columns: list = None,
                                   plots_per_page: int = 6,
                                   figsize: tuple = (15, 10),
                                   save_pdf: str = None) -> dict:
        """
        Plot distributions of numeric columns with multiple plots per page and KDE.
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
                    
                # Create subplot grid
                fig, axes = plt.subplots(plots_per_page // 2, 2, figsize=figsize)
                axes = axes.ravel()
                    
                for idx, col in enumerate(batch_cols):
                    ax = axes[idx]
                    data = df[col]
                        
                    # Calculate statistics
                    stats = self._calculate_statistics(data)
                    stats_dict[col] = stats
                        
                    # Create histogram with KDE
                    sns.histplot(data=data, ax=ax, kde=True, 
                                 color='skyblue', edgecolor='black',
                                 alpha=0.6, bins=30)
                        
                    # Add mean and median lines
                    ax.axvline(stats['mean'], color='red', linestyle='--', 
                               label=f"Mean: {stats['mean']:.2f}")
                    ax.axvline(stats['50%'], color='green', linestyle='--', 
                               label=f"Median: {stats['50%']:.2f}")
                        
                    # Customize plot
                    ax.set_title(f'Distribution of {col}')
                    ax.legend(fontsize='small')
                    ax.grid(True, alpha=0.3)
                        
                    # Add statistics text
                    stats_text = (
                        f"Count: {stats['count']:.0f}\n"
                        f"Mean: {stats['mean']:.2f}\n"
                        f"Std: {stats['std']:.2f}\n"
                        f"Skew: {stats['skewness']:.2f}\n"
                        f"Kurt: {stats['kurtosis']:.2f}\n"
                        f"Missing: {stats['missing']:.0f}\n"
                        f"Missing %: {stats['missing_pct']:.1f}%"
                    )
                        
                    ax.text(0.95, 0.95, stats_text,
                            transform=ax.transAxes,
                            verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(facecolor='white', alpha=0.8),
                            fontsize='small')
                    
                # Hide empty subplots
                for idx in range(len(batch_cols), len(axes)):
                    axes[idx].set_visible(False)
                    
                # Adjust layout
                plt.tight_layout()
                    
                if pdf_handle:
                    pdf_handle.savefig(fig, dpi=self.dpi, bbox_inches='tight')
                    
                plt.show()
                plt.close()
                    
                print(f"Processed figure {fig_num + 1} of {total_figures}")
                
            # Print statistics to terminal
            self._print_statistics(stats_dict)
                
            # Save statistics
            if save_pdf:
                stats_filename = save_pdf.replace('.pdf', '_stats')
                self._save_statistics(stats_dict, stats_filename)
                    
        finally:
            if pdf_handle:
                pdf_handle.close()
                print(f"\nPlots saved to: {pdf_path}")
                    
        return stats_dict

# Example usage with different types of distributions
if __name__ == "__main__":
    # Create sample data with various distributions
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'Normal': np.random.normal(0, 1, n_samples),
        'Skewed': np.random.exponential(2, n_samples),
        'Bimodal': np.concatenate([
            np.random.normal(-2, 0.5, n_samples//2),
            np.random.normal(2, 0.5, n_samples//2)
        ]),
        'Uniform': np.random.uniform(-3, 3, n_samples),
        'LogNormal': np.random.lognormal(0, 1, n_samples),
        'ChiSquare': np.random.chisquare(5, n_samples)
    })
    
    # Initialize visualizer
    viz = DataVisualizer(output_dir='visualization_output')
    
    # Create plots
    stats = viz.plot_numeric_distributions(
        df,
        plots_per_page=6,
        save_pdf='distributions.pdf'
    )
