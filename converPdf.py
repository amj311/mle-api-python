import subprocess

all_pdfs_directory = 'static'
pdf_name = "test_pdf"
pdf_directory = all_pdfs_directory+"/"+pdf_name
pdf_path = pdf_directory+"/original.pdf"
out_path = pdf_directory+"/img"
subprocess.call(["pdftoppm", pdf_path, out_path, "-png", "-scale-to", "2400"])