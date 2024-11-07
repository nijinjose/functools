import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import seaborn as sns
import warnings

# Suppress warnings from seaborn/matplotlib
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
        else:
            return super().default(obj)

class DataVisualizer:
    """
    A class for visualizing and analyzing numerical data from DataFrames.
    Creates distribution plots and saves statistics.
    """
    
    def __init__(self, output_dir=None, dpi=300):
        """
        Initialize the DataVisualizer.
        
        Args:
            output_dir (str): Directory to save outputs.
            dpi (int): DPI for saved plots.
        """
        self.dpi = dpi
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _calculate_statistics(self, data: pd.Series) -> dict:
        """Calculate comprehensive statistics for a series."""
        stats = data.describe().to_dict()
        stats['skewness'] = data.skew()
        stats['kurtosis'] = data.kurtosis()
        stats['missing'] = data.isnull().sum()
        stats['missing_pct'] = data.isnull().mean() * 100
        stats['unique'] = data.nunique()
        return {k: float(v) for k, v in stats.items()}
    
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
        Plot distributions of numeric columns and calculate statistics.
        
        Args:
            df: Input DataFrame.
            columns: Specific columns to plot (if None, uses all numeric).
            plots_per_page: Number of plots per page.
            figsize: Figure size in inches.
            save_pdf: Filename for saving PDF.
                
        Returns:
            Dictionary containing statistics for each column.
        """
        numeric_cols = columns or df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in the dataset.")
        print(f"\nAnalyzing {len(numeric_cols)} numeric columns...")

        stats_dict = {}
        pdf_path = self.output_dir / save_pdf if save_pdf else None
        pdf_handle = PdfPages(pdf_path) if pdf_path else None
        
        try:
            for idx, col in enumerate(numeric_cols):
                data = df[col]
                stats = self._calculate_statistics(data)
                stats_dict[col] = stats

                plt.figure(figsize=figsize)
                sns.histplot(data, kde=True, bins=30, color='skyblue', edgecolor='black')
                plt.title(f'Distribution of {col}')
                plt.axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.2f}")
                plt.axvline(stats['50%'], color='green', linestyle='--', label=f"Median: {stats['50%']:.2f}")
                plt.legend()

                # Annotate statistics
                stats_text = '\n'.join([
                    f"Count: {stats['count']:.0f}",
                    f"Mean: {stats['mean']:.2f}",
                    f"Std: {stats['std']:.2f}",
                    f"Skewness: {stats['skewness']:.2f}",
                    f"Kurtosis: {stats['kurtosis']:.2f}",
                    f"Missing: {stats['missing']:.0f} ({stats['missing_pct']:.1f}%)"
                ])
                plt.gca().text(0.95, 0.95, stats_text,
                               transform=plt.gca().transAxes,
                               verticalalignment='top',
                               horizontalalignment='right',
                               bbox=dict(facecolor='white', alpha=0.6))

                if pdf_handle:
                    pdf_handle.savefig(dpi=self.dpi)
                plt.show()
                plt.close()
            
            # Save statistics
            if save_pdf:
                stats_filename = save_pdf.replace('.pdf', '_stats')
                self._save_statistics(stats_dict, stats_filename)
        finally:
            if pdf_handle:
                pdf_handle.close()
                print(f"\nPlots saved to: {pdf_path}")
        return stats_dict

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'Normal': np.random.normal(0, 1, 1000),
        'Skewed': np.random.exponential(2, 1000),
        'High_Kurtosis': np.random.standard_t(3, 1000),
        'Uniform': np.random.uniform(0, 10, 1000)
    })
    
    # Initialize visualizer
    viz = DataVisualizer(output_dir='visualization_output')
    
    # Create plots and get statistics
    stats = viz.plot_numeric_distributions(
        df,
        save_pdf='distributions.pdf'
    )
