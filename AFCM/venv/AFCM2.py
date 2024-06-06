from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import requests
import io
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import os
import joblib
from datetime import datetime
from glob import glob
import subprocess
from decouple import config
import json
import yake
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

app = Flask(__name__)
CORS(app)

# Load the OpenAI API key from the .env file
openai_api_key = config('OPENAI_API_KEY')

# Set the API key for the OpenAI client
openai.api_key = openai_api_key

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
    
def process_text_in_chunks(text, max_tokens=3400, context_window_size=400):
    # Split the text into paragraphs
    paragraphs = text.split('\n')
    first_paragraph = paragraphs[0]

    # Tokenize the text and remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Split the tokens into chunks with a context window
    chunks = [tokens[max(0, i-context_window_size):i+max_tokens] for i in range(0, len(tokens), max_tokens)]

    # Process each chunk separately and combine the results
    combined_result = ''
    for chunk in chunks:
        # Include the first paragraph as context
        chunk_text = first_paragraph + ' ' + ' '.join(chunk)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": chunk_text}
            ]
        )
        combined_result += response['choices'][0]['message']['content']

    return combined_result

# Function to remove stop words
def remove_stop_words(text):
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.lower() not in stop_words]
    processed_text = ' '.join(words)
    return processed_text

def remove_links(text):
    # Remove links using a regular expression
    return re.sub(r'http\S+', '', text)

def summarize_pdf(pdf_url):
    pdf_content = download_pdf(pdf_url)

    if pdf_content is not None:
        try:
            with io.BytesIO(pdf_content) as pdf_stream:
                with pdfplumber.open(pdf_stream) as pdf:
                    pdf_text = " ".join(page.extract_text() for page in pdf.pages)
        except pdfplumber.PDFSyntaxError:
            print(f"Failed to open PDF from {pdf_url}: Not a valid PDF")
            return None

        # Remove links
        pdf_text = remove_links(pdf_text)

        # Use sumy for summarization
        parser = PlaintextParser.from_string(pdf_text, Tokenizer("english"))
        summarizer = LsaSummarizer()

        # Summarize the document with a dynamic number of sentences (e.g., 5 or 6)
        num_sentences = min(6, len(parser.document.sentences))
        summary = summarizer(parser.document, num_sentences)

        return " ".join(str(sentence) for sentence in summary)
    else:
        return None

# Load the latest machine learning model (pipeline)
existing_models = glob(os.path.join(os.path.dirname(__file__), 'model', 'AFCM_pipeline*.joblib'))
latest_counts = [int(re.search(r'\d+', model).group()) for model in existing_models if re.search(r'\d+', model) is not None]
latest_count = max(latest_counts, default=0)
latest_model_path = os.path.join(os.path.dirname(__file__), 'model', f'AFCM_pipeline{latest_count}.joblib')
pipeline = joblib.load(latest_model_path, mmap_mode='r')

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

def remove_stops(text):
    text = re.sub(r'M\d+_GADD\d+_\d+_SE_C01\.QXD \d+/\d+/\d+ \d+:\d+ [APMapm]{2} Page \d+', '', text)
    text = text.replace("\n", "")

    return text.strip()

def train_svm_model():
    try:
        current_date = datetime.now()
        if current_date.month == 12 and current_date.day == 20:
            # Send a GET request to the Laravel application
            response = requests.get('https://gener-lnulib.site/api/resources')

            # Check if the request was successful
            if response.status_code == 200:
                # Convert the response to JSON
                resources = response.json()

                # Find all existing JSON files
                existing_files = glob("/home/gener/afcm/training_data/resources*.json")

                # Extract the counts from the filenames
                counts = [int(re.search(r'\d+', file).group()) for file in existing_files if re.search(r'\d+', file) is not None]

                # Find the maximum count and increment it for the new file
                count = max(counts, default=0) + 1

                # Write the resources to a new JSON file with the incremented count
                with open(f'resources{count}.json', 'w') as f:
                    json.dump(resources, f)

                svm_script_path = '/home/gener/afcm/retraining.py'
                subprocess.run(['python', svm_script_path])
                print("SVM model re-training completed successfully.")
            else:
                print("Failed to get resources from Laravel application.")
        else:
            print("SVM model training skipped. Today is not the 20th of December.")
    except Exception as e:
        print(f"Error during SVM model training: {e}")

# Train the SVM model only on the 20th of December
train_svm_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
