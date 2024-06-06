from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pdfplumber
import os
import re
from joblib import dump, load

# Download NLTK data for tokenization (if not already downloaded)
# nltk.download('punkt')

# Flatten the DataFrame and convert to a list of strings
df = pd.read_excel("keyterms_validation.xlsx")

# Initialize an empty dictionary to store the key terms by discipline
keyterms_by_discipline = {}

# Iterate through the columns of the DataFrame (assuming each column represents a discipline)
for column in df.columns:
    # Extract the key terms for the current discipline and convert them to a list of strings
    keyterms = df[column].dropna().astype(str).tolist()
    
    # Store the key terms in the dictionary with the discipline as the key
    keyterms_by_discipline[column] = keyterms

# Extract text from PDFs

# Define the folder containing the PDFs
folder_path = "C:/Users/LENOVO/Desktop/python_scripts/resources"  # Replace with the path to your folder

# Initialize a list to store text content
text_data = []

# List all PDF files in the folder
pdf_filenames = [filename for filename in os.listdir(folder_path) if filename.endswith(".pdf")]

# Iterate through PDF files and extract text
for filename in pdf_filenames:
    pdf_file_path = os.path.join(folder_path, filename)
    with pdfplumber.open(pdf_file_path) as pdf:
        page_text = ""
        for page in pdf.pages:
            page_text += page.extract_text()
        text_data.append(page_text)

# Now, text_data contains the text content of each PDF in the "resources" folder.

tokens = []

for text in text_data:
    tokens.extend(word_tokenize(text))

# Lowercasing
tokens = [word.lower() for word in tokens]

# Stopword Removal
#nltk.download('stopwords') #stopwords package is for removing stopwords
stop_words = set(stopwords.words('english')) #set of stopwords in english
filtered_tokens = [word for word in tokens if word not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

# Lemmatization
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# Remove punctuation and non-alphanumeric characters
cleaned_tokens = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in lemmatized_tokens]

# Join tokens back into text
preprocessed_text = ' '.join(cleaned_tokens)

# Implementing TF-IDF

# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text])

# Implementing Word2Vec

tokenized_text = preprocessed_text.split()
model = Word2Vec([tokenized_text], vector_size=100, window=5, min_count=1, sg=0)

key_terms = []

# Iterate through the columns of the DataFrame
for column in df.columns:
    key_terms.extend(df[column].dropna().astype(str).tolist())
    
key_term_embeddings = {}  # To store embeddings for each key term

for term in key_terms:
    if term in model.wv:
        key_term_embeddings[term] = model.wv[term]
    else:
        # Handle terms that are not in the vocabulary (out of vocabulary words)
        key_term_embeddings[term] = None  # You can use None or zeros, depending on your choice

def assign_label_function(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    # Extract the file name from the file path
    file_name = os.path.basename(file_path)
    
    # Extract the discipline from the file name (assuming the file name is formatted as 'discipline_file.pdf')
    discipline = file_name.split('_')[0]
    
    # Return the discipline as the label
    return discipline

# Define X and y
X = []  # This should contain embeddings for all PDFs
y = []  # This should contain corresponding discipline labels

# Iterate through PDFs and compute embeddings
for pdf_text, pdf_filename in zip(text_data, pdf_filenames):
    pdf_file_path = os.path.join(folder_path, pdf_filename)
    pdf_embedding = np.mean([model.wv[word] for word in pdf_text.split() if word in model.wv], axis=0)
    X.append(pdf_embedding)
    # Assign a label based on the file name or some other criterion
    # Replace 'assign_label_function' with your logic for assigning labels
    y.append(assign_label_function(pdf_file_path))

# Create discipline embeddings (you can use the same logic as for key terms)
discipline_embeddings = {}  # To store embeddings for each discipline
for discipline, terms in keyterms_by_discipline.items():
    discipline_embedding = np.mean([model.wv[term] for term in terms if term in model.wv], axis=0)
    discipline_embeddings[discipline] = discipline_embedding
    
# Debugging: Print the unique labels in y
unique_labels = set(y)
print("Unique Labels:", unique_labels)

# Check if there are at least two unique labels/classes
if len(unique_labels) < 2:
    raise ValueError("There are not enough unique classes for classification.")

# Implementing Support Vector Machines

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use an SVM with RBF kernel
classifier = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)

classifier.fit(X_train, y_train)

pdf_embedding = np.mean([model.wv[word] for word in pdf_text.split() if word in model.wv], axis=0)

# Find the most similar discipline
similarity_scores = {
    discipline: np.dot(pdf_embedding, discipline_embedding)
    for discipline, discipline_embedding in discipline_embeddings.items()
}

# Assign the PDF to the discipline with the highest similarity
assigned_discipline = max(similarity_scores, key=similarity_scores.get)

for pdf_filename, assigned_discipline in zip(pdf_filenames, y):
    print(f"PDF File: {pdf_filename}, Assigned Discipline: {assigned_discipline}")
    
# Save the trained SVM model
dump(classifier, "svm_model.joblib")

# Save the Word2Vec model to a file
model.save("C:/Users/LENOVO/Desktop/python_scripts/word2vec_model.bin")

### IMPLEMENT THE OPTIMIZATION AND PARALLEL PROCESSING MODIFICATIONS AS PROVIDED BY NOIR PLEASE THANK YOU ###
