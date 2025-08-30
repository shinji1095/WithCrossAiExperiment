from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def export_all_figures_to_pdf(pdf_path):
    pdf = PdfPages(pdf_path)
    figures = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figures:
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()