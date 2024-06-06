from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import joblib
from glob import glob
import io
import pdfplumber
import re
import json
from datetime import datetime, time
import subprocess
import shutil
import time as t
import threading
import schedule

app = Flask(__name__)
CORS(app)

# Function to download a PDF document
def download_pdf(pdf_url):
    try:
        # Check if the URL is a Google Drive link
        if 'drive.google.com' in pdf_url:
            # Extract the file ID from the URL
            file_id = pdf_url.split('=')[-1]
            # Create a direct download link
            pdf_url = f'https://drive.google.com/uc?export=download&id={file_id}'

        response = requests.get(pdf_url)
        response.raise_for_status()
        if 'application/pdf' not in response.headers['content-type']:
            print(f"URL does not point to a PDF: {pdf_url}")
            return None
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF from {pdf_url}: {str(e)}")
        return None

# Load the latest machine learning model (pipeline)
existing_models = glob(os.path.join(os.path.dirname(__file__), 'model', 'AFCM_pipeline*.joblib'))
latest_counts = [int(re.search(r'\d+', model).group()) for model in existing_models if re.search(r'\d+', model) is not None]
latest_count = max(latest_counts, default=0)
latest_model_path = os.path.join(os.path.dirname(__file__), 'model', f'AFCM_pipeline{latest_count}.joblib')

# Mapping of disciplines to colleges
discipline_to_college = {
    'Computer Science': 'CAS',
    'Mathematics': 'CAS',
    'Natural Sciences': 'CAS',
    'The Arts': 'CAS',
    'Applied Sciences': 'CAS',
    'Social Sciences': 'CAS',
    'Language': 'CME',
    'Linguistics': 'CME',
    'Literature': 'CME',
    'Geography': 'CME',
    'Management': 'CME',
    'Philosophy': 'COE',
    'Psychology': 'COE',
    'History': 'COE',
}

# Global variable to hold the model
afcm_model = None

def get_model():
    global afcm_model
    if afcm_model is None:
        afcm_model = joblib.load(latest_model_path, mmap_mode='c')
    return afcm_model

@app.route('/autofill', methods=['POST'])
def autofill():
    if request.method == 'POST':
        if 'url' in request.json:
            pdf_url = request.json['url']

            # Download the PDF file using the URL
            pdf_content_bytes = download_pdf(pdf_url)

            if pdf_content_bytes is not None:
                pdf_content_text = ""
                with io.BytesIO(pdf_content_bytes) as pdf_buffer:
                    for page in pdfplumber.open(pdf_buffer).pages:
                        pdf_content_text += page.extract_text()

                pipeline = get_model()
                predicted_discipline = pipeline.predict([pdf_content_text])
                college = discipline_to_college.get(predicted_discipline[0], 'Unknown')

                return jsonify({'discipline': predicted_discipline[0], 'college': college})

    return jsonify({'error': 'Invalid PDF URL'})

def train_svm_model():
    try:
        current_date = datetime.now()
        if current_date.day == 28:  # Check if it's the 28th
            # Send a GET request to the Laravel application
            response = requests.get('https://gener-lnulib.site/api/resources')

            # Check if the request was successful
            if response.status_code == 200:
                # Convert the response to JSON
                resources = response.json()

                # Directory where the AI script keeps the initial training data
                training_data_dir = "C:/Users/LENOVO/Desktop/python_scripts/test_folder"

                # Download the JSON files from each URL and save them in the training data directory
                for resource in resources:
                    json_url = resource['json_url']
                    response = requests.get(json_url, stream=True)
                    # Get the filename from the Content-Disposition header
                    content_disposition = response.headers.get('content-disposition')
                    if content_disposition:
                        filename = re.findall('filename="(.+)"', content_disposition)[0]
                        filename = os.path.join(training_data_dir, filename)
                        with open(filename, 'wb') as f:
                            shutil.copyfileobj(response.raw, f)
            
                svm_script_path = 'C:/Users/LENOVO/Desktop/afcmflask/AFCM/venv/retraining.py'
                subprocess.run(['python', svm_script_path])
                print("SVM model re-training completed successfully.")
            else:
                print("Failed to get resources from Laravel application.")
        else:
            print("SVM model training skipped. Today is not the 28th.")
    except Exception as e:
        print(f"Error during SVM model training: {e}")

# Train the SVM model only on the 20th of December
train_svm_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)