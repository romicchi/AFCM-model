import os
import pdfplumber
import numpy as np
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from joblib import load
import datetime

# Load the trained SVM model
model_path = "svm_model.joblib"
classifier = load(model_path)

# Load the trained Word2Vec model
model = Word2Vec.load("C:/Users/LENOVO/Desktop/python_scripts/word2vec_model.bin")

# Get the list of PDF files in the resources folder
pdf_files = [f for f in os.listdir('resources') if f.endswith('.pdf')]

# Display the list of PDF files
print("PDF files in the resources folder:")
for i, pdf_file in enumerate(pdf_files):
    print(f"{i+1}. {pdf_file}")

# Prompt the user to select a PDF file
pdf_file_index = int(input("Enter the index of the PDF file to analyze: ")) - 1
pdf_file_name = os.path.join('resources', pdf_files[pdf_file_index])

# Read the content of the PDF file
with pdfplumber.open(pdf_file_name) as pdf:
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

# Perform text preprocessing
tokens = []

for text in word_tokenize(pdf_text):
    tokens.extend(word_tokenize(text))

# Lowercasing
tokens = [word.lower() for word in tokens]

# Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Remove punctuation and non-alphanumeric characters
cleaned_tokens = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in filtered_tokens]

# Join tokens back into text
preprocessed_text = ' '.join(tokens)

# Generate embeddings for the PDF text
pdf_embedding = np.mean([model.wv[word] for word in preprocessed_text.split() if word in model.wv.key_to_index], axis=0)

# Now, make predictions
predicted_discipline = classifier.predict([pdf_embedding])[0]

# Print the predicted discipline
print("Predicted Discipline:", predicted_discipline)