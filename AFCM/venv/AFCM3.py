from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import io
import pdfplumber
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os
import joblib
from glob import glob
import json
import yake
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)
CORS(app)

# Function to download a PDF document
def download_pdf(pdf_url):
    try:
        if 'drive.google.com' in pdf_url:
            file_id = pdf_url.split('=')[-1]
            pdf_url = f'https://drive.google.com/uc?export=download&id={file_id}'

        print(f"Downloading PDF from {pdf_url}")
        response = requests.get(pdf_url)
        response.raise_for_status()

        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")

        if 'application/pdf' not in response.headers['content-type'] and 'application/octet-stream' not in response.headers['content-type']:
            print(f"URL does not point to a PDF: {pdf_url}")
            return None

        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF from {pdf_url}: {str(e)}")
        return None

# Function to remove stop words
def remove_stop_words(text):
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.lower() not in stop_words]
    processed_text = ' '.join(words)
    return processed_text

def extract_text_from_pdf(pdf_content):
    with io.BytesIO(pdf_content) as pdf_stream:
        with pdfplumber.open(pdf_stream) as pdf:
            for page in pdf.pages:
                yield page.extract_text()

# Use T5 for summarization
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def summarize_pdf(pdf_url):
    pdf_content = download_pdf(pdf_url)

    if pdf_content is not None:
        try:
            pdf_text = " ".join(extract_text_from_pdf(pdf_content))
        except pdfplumber.PDFSyntaxError:
            print(f"Failed to open PDF from {pdf_url}: Not a valid PDF")
            return None

        inputs = tokenizer.encode("summarize: " + pdf_text[:512], return_tensors="pt", truncation=True)
        outputs = model.generate(inputs, max_length=200, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

        summary = tokenizer.decode(outputs[0]).replace('<pad>', '').replace('</s>', '').strip()
        summary = summary[0].upper() + summary[1:]

        return summary
    else:
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

                cleaned_text = preprocess_text(remove_stop_words(pdf_content_text))
                keywords = extract_keywords(cleaned_text)
                summary = summarize_pdf(pdf_url)

                pipeline = get_model()
                predicted_discipline = pipeline.predict([cleaned_text])
                college = discipline_to_college.get(predicted_discipline[0], 'Unknown')

                return jsonify({'discipline': predicted_discipline[0], 'college': college, 'keywords': keywords, 'summary': summary})

    return jsonify({'error': 'Invalid PDF URL'})

def preprocess_text(text):
    stop_words_custom = set(["would", "could", "should", "might", "must", "shall"])
    stop_words = stop_words_custom.union(set(stopwords.words('english')))
    
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words]

    return ' '.join(tokens)

def extract_keywords(text, max_keywords=5):
    # Initialize a YAKE keyword extractor
    kw_extractor = yake.KeywordExtractor(lan="en")

    # Remove stop words from the text
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])

    # Extract keywords
    keywords = kw_extractor.extract_keywords(text)

    # Split the phrases into individual words and keep their scores
    keywords = [(word, score) for keyword, score in keywords for word in keyword.split()]

    # Sort the words by their scores in descending order
    keywords.sort(key=lambda x: x[1], reverse=True)

    # Get the top words
    top_keywords = [word for word, score in keywords[:max_keywords]]

    return top_keywords

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
