from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import io
import pdfplumber
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize

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

def extract_text_from_pdf(pdf_content):
    with io.BytesIO(pdf_content) as pdf_stream:
        with pdfplumber.open(pdf_stream) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                # Remove the first and last line (header and footer)
                lines = text.split('\n')
                if len(lines) > 2:
                    text = '\n'.join(lines[1:-1])
                yield text

# Use T5 for summarization
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def summarize_pdf(pdf_url):
    pdf_content = download_pdf(pdf_url)
    pdf_text = " ".join(extract_text_from_pdf(pdf_content))

    # Split the text into paragraphs
    paragraphs = pdf_text.split('\n')

    # Summarize each paragraph separately
    summaries = []
    for paragraph in paragraphs:
        inputs = tokenizer.encode("summarize: " + paragraph, return_tensors="pt", truncation=True)
        outputs = model.generate(inputs, max_length=200, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0]).replace('<pad>', '').replace('</s>', '').strip()
        summaries.append(summary)

    # Join the summaries back together
    summary = ' '.join(summaries)
    summary = summary[0].upper() + summary[1:]

    return summary

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

                summary = summarize_pdf(pdf_url)

                return jsonify({'summary': summary})

    return jsonify({'error': 'Invalid PDF URL'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)