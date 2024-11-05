import matplotlib.pyplot as plt  # For creating plots
from matplotlib.backends.backend_pdf import PdfPages  # For saving plots to PDF
import pandas as pd  # For handling DataFrames
import numpy as np  # For numerical operations
from typing import List, Optional, Tuple, Union  # For type hints
import gc  # For memory management

class DataFramePlotSaver:
    """
    A class for saving DataFrame plots to PDF files.
    Think of it like a digital photo album where each page can contain multiple plots.
    """
    
    def __init__(self, 
                 pdf_filename: str = 'distribution_plots.pdf',  # Name of output file
                 plots_per_page: int = 6,  # How many plots on each page
                 figsize: Tuple[float, float] = (8.27, 11.69),  # Page size (A4)
                 dpi: int = 300):  # Image quality
        """
        Sets up the initial configuration for saving plots.
        
        Parameters:
        - pdf_filename: What to name your PDF file
        - plots_per_page: How many plots to put on each page
        - figsize: How big to make each page (width, height) in inches
        - dpi: Image quality (higher = better quality but larger file)
        """
        self.pdf_filename = pdf_filename
        self.plots_per_page = plots_per_page
        self.figsize = figsize
        self.dpi = dpi
        self.pdf = None  # Will hold our PDF file
        self.plot_count = 0  # Keeps track of how many plots we've made
    
    def __enter__(self):
        """
        This is called when you use 'with' statement.
        It opens the PDF file for writing.
        """
        self.pdf = PdfPages(self.pdf_filename)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        This is called when the 'with' block ends.
        It closes the PDF file and prints a summary.
        """
        if self.pdf:
            self.pdf.close()
            print(f"Saved {self.plot_count} plots to '{self.pdf_filename}'")
    
    def save_histograms(self, 
                       df: pd.DataFrame,  # Your data
                       columns: Optional[List[str]] = None,  # Which columns to plot
                       bins: Union[int, str] = 30,  # How many bars in histogram
                       include_stats: bool = True,  # Whether to show statistics
                       progress_bar: bool = True) -> None:  # Whether to show progress
        """
        Creates histograms for numeric columns in your DataFrame.
        
        Parameters:
        - df: Your pandas DataFrame
        - columns: Which columns to plot (if None, uses all numeric columns)
        - bins: How many bars to use in histogram
        - include_stats: Whether to show mean, median, etc.
        - progress_bar: Whether to show progress updates
        """
        # If no columns specified, find all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Set up the grid layout (2 columns per page)
        n_cols = 2
        n_rows = self.plots_per_page // n_cols  # How many rows we need
        
        # Process columns in groups (one group = one page)
        for i in range(0, len(columns), self.plots_per_page):
            # Get the columns for this page
            batch_columns = columns[i:i + self.plots_per_page]
            
            # Create a new figure (page) with subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
            axes = axes.ravel()  # Convert 2D array of plots to 1D for easier handling
            
            # Create histograms for each column in this batch
            for idx, col in enumerate(batch_columns):
                if idx < len(axes):
                    ax = axes[idx]
                    data = df[col].dropna()  # Remove missing values
                    
                    if len(data) > 0:
                        # Create the histogram
                        ax.hist(data, bins=bins, edgecolor='black')
                        ax.set_title(f'Distribution of {col}')
                        ax.set_xlabel(col)
                        ax.set_ylabel('Frequency')
                        ax.grid(axis='y', linestyle='--')
                        
                        # Add statistical information if requested
                        if include_stats:
                            stats_text = (
                                f'Mean: {data.mean():.2f}\n'  # Average
                                f'Median: {data.median():.2f}\n'  # Middle value
                                f'Std: {data.std():.2f}\n'  # Spread
                                f'N: {len(data)}'  # Number of data points
                            )
                            # Add stats text to top right corner
                            ax.text(0.95, 0.95, stats_text,
                                  transform=ax.transAxes,
                                  verticalalignment='top',
                                  horizontalalignment='right',
                                  bbox=dict(facecolor='white', alpha=0.8))
            
            # Hide any empty subplot spaces
            for idx in range(len(batch_columns), len(axes)):
                axes[idx].set_visible(False)
            
            # Adjust layout and save
            plt.tight_layout()
            self.pdf.savefig(fig, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
            self.plot_count += 1
            
            # Show progress if requested
            if progress_bar:
                processed = min(i + self.plots_per_page, len(columns))
                print(f"Processed {processed}/{len(columns)} columns")
            
            # Clean up memory
            gc.collect()

# Example of how to use the class:
def example_usage():
    """
    Shows how to use the DataFramePlotSaver class.
    """
    # Create some example data
    df = pd.DataFrame({
        'Height': np.random.normal(170, 10, 1000),  # Normal distribution
        'Weight': np.random.normal(70, 15, 1000),   # Normal distribution
        'Age': np.random.uniform(20, 80, 1000),     # Uniform distribution
    })
    
    # Use the class to save plots
    with DataFramePlotSaver('my_plots.pdf', plots_per_page=4) as saver:
        saver.save_histograms(df, include_stats=True)

# Run the example if this file is run directly
if __name__ == '__main__':
    example_usage()
