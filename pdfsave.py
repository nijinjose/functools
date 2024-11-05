# pdf_saver.py

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class PDFSaver:
    def __init__(self, pdf_filename='plots.pdf'):
        """
        Initializes the PDFSaver class with a PDF file to save plots into.

        Parameters:
        pdf_filename (str): Name of the PDF file where plots will be saved.
        """
        self.pdf_filename = pdf_filename
        self.pdf = PdfPages(self.pdf_filename)

    def save_plot(self, plot_func, *args, **kwargs):
        """
        Executes a plotting function and saves the resulting plot to the PDF.

        Parameters:
        plot_func (function): A function to create the plot (e.g., `plt.plot`).
        *args, **kwargs: Arguments to pass to the plotting function.
        """
        plot_func(*args, **kwargs)  # Creates the plot with provided arguments
        self.pdf.savefig()  # Saves the current plot to the PDF
        plt.close()  # Closes the current plot to free up memory

    def close(self):
        """
        Closes the PDF file and finalizes it.
        """
        self.pdf.close()
        print(f"All plots saved to '{self.pdf_filename}'")
