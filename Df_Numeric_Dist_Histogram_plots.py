import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import gc
from typing import Optional, List, Union, Dict
from pathlib import Path
import json
from datetime import datetime
import sys
from tabulate import tabulate

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DataVisualizer:
    def __init__(self, 
                 output_dir: Optional[str] = None,
                 dpi: int = 300):
        self.dpi = dpi
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):  # Handle NaN/None values
            return None
        return obj
    
    def _save_statistics(self, stats: Dict, filename: str) -> None:
        """Save statistics to JSON file with proper type conversion."""
        file_path = self.logs_dir / f"{filename}_{self.timestamp}.json"
        
        # Convert all numpy types to Python native types
        serializable_stats = self._convert_to_serializable(stats)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_stats, f, indent=4, cls=NumpyEncoder)
        print(f"\nStatistics saved to: {file_path}")
    
    def _get_statistics(self, data: pd.Series) -> Dict:
        """Calculate statistics with proper type conversion."""
        stats = {
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'skew': float(data.skew()),
            'kurtosis': float(data.kurtosis()),
            'missing': int(data.isnull().sum()),
            'missing_pct': float((data.isnull().sum() / len(data)) * 100),
            'unique_values': int(data.nunique()),
            'min': float(data.min()),
            'max': float(data.max())
        }
        return stats

    def plot_numeric_distributions(self, 
                                 df: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 plots_per_page: int = 6,
                                 figsize: tuple = (15, 10),
                                 save_pdf: Optional[str] = None) -> Dict[str, Dict]:
        """Plot distributions with proper type handling."""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist() \
                      if columns is None else \
                      [col for col in columns if df[col].dtype in ['int64', 'float64']]
        
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
                    data = df[col].dropna()
                    
                    # Create histogram
                    ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
                    
                    # Add mean and median lines
                    mean_val = float(data.mean())
                    median_val = float(data.median())
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                             label=f'Mean: {mean_val:.2f}')
                    ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, 
                             label=f'Median: {median_val:.2f}')
                    
                    ax.set_title(f'Distribution of {col}')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Calculate statistics
                    stats = self._get_statistics(df[col])
                    stats_dict[col] = stats
                    
                    # Add stats to plot
                    stats_text = "\n".join([f"{k}: {v:.2f}" if isinstance(v, float)
                                          else f"{k}: {v}" 
                                          for k, v in stats.items()])
                    ax.text(0.95, 0.95, stats_text,
                           transform=ax.transAxes,
                           verticalalignment='top',
                           horizontalalignment='right',
                           bbox=dict(facecolor='white', alpha=0.8))
                
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
        """Plot correlation heatmap with proper type handling."""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist() \
                      if columns is None else \
                      [col for col in columns if df[col].dtype in ['int64', 'float64']]
        
        print(f"\nCalculating correlations for {len(numeric_cols)} columns...")
        corr_matrix = df[numeric_cols].corr()
        
        if min_correlation > 0:
            mask = np.abs(corr_matrix) >= min_correlation
            corr_matrix = corr_matrix.where(mask, 0)
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im)
        
        ax.set_xticks(np.arange(len(numeric_cols)))
        ax.set_yticks(np.arange(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax.set_yticklabels(numeric_cols)
        
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha='center', va='center')
        
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if save_pdf:
            pdf_path = self.output_dir / save_pdf
            plt.savefig(pdf_path, bbox_inches='tight', dpi=self.dpi)
            print(f"\nCorrelation heatmap saved to: {pdf_path}")
            
            # Save correlation matrix
            matrix_filename = save_pdf.replace('.pdf', '_matrix')
            corr_dict = self._convert_to_serializable(corr_matrix.to_dict())
            self._save_statistics(corr_dict, matrix_filename)
        
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
    
    # Plot and save distributions
    stats = viz.plot_numeric_distributions(
        df,
        save_pdf='distributions.pdf'
    )
    
    # Plot and save correlations
    corr_matrix = viz.plot_correlation_heatmap(
        df,
        save_pdf='correlations.pdf'
    )
