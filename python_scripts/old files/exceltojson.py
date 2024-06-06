import os
import ujson
import pdfplumber
from concurrent.futures import ThreadPoolExecutor

# Function to convert PDF to JSON using pdfplumber
def pdf_to_json(pdf_path, json_file_path):
    with pdfplumber.open(pdf_path) as pdf, open(json_file_path, 'w') as json_file:
        # Initialize an empty string to hold the text of each page
        pdf_content = ""

        # Process each page individually
        for page in pdf.pages:
            page_text = page.extract_text()
            # Append the current page's text to the string, with a space in between
            pdf_content += page_text + " "

        # Write the text content to the JSON file
        json_file.write(ujson.dumps({"text": pdf_content}))

# Input and output directories
pdf_folder = 'C:/Users/LENOVO/Desktop/python_scripts/old files/resources'
json_folder = 'C:/Users/LENOVO/Desktop/python_scripts/json_resources'

# Create the output folder if it doesn't exist
if not os.path.exists(json_folder):
    os.makedirs(json_folder)

def process_pdf(pdf_file):
    pdf_path = os.path.join(pdf_folder, pdf_file)

    # Create a corresponding JSON file in the output folder
    json_file_path = os.path.join(json_folder, os.path.splitext(pdf_file)[0] + '.json')

    # Convert PDF to JSON using pdfplumber
    pdf_to_json(pdf_path, json_file_path)

# Use ThreadPoolExecutor for concurrent processing
with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    executor.map(process_pdf, pdf_files)

print("Conversion complete. JSON files are saved in the 'json_resources' folder.")