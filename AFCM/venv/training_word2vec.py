import joblib
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from glob import glob
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string
import json
import os
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(folder):
    all_data = []

    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r', encoding="utf-8") as f:
                data = json.load(f)
                # Extract discipline information from the filename
                discipline = re.search(r'^(.*?)_', filename).group(1)
                data['discipline'] = discipline
                all_data.append(data)

    return all_data

def write_data(file, data):
    with open(file, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def remove_stops(text):
    # Use regular expression to match and remove the dynamic pattern
    text = re.sub(r'M\d+_GADD\d+_\d+_SE_C01\.QXD \d+/\d+/\d+ \d+:\d+ [APMapm]{2} Page \d+', '', text)

    # Remove line breaks
    text = text.replace("\n", "")

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove digits
    text = "".join([i for i in text if not i.isdigit()])
    
    # Remove specific URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Perform lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return words

def clean_docs(docs):
    final = []
    for doc in docs:
        if 'text' in doc:
            clean_doc = remove_stops(doc['text'])
            # Add further processing here
            final.append(clean_doc)
        else:
            print(f"Warning: 'text' key not found in document: {doc}")
    return final

# Load Data
descriptions = load_data("C:/Users/LENOVO/Desktop/python_scripts/json_resources")

# Cleaning the Data
cleaned_docs = clean_docs(descriptions)

# Define the Word2Vec model
w2v_model = Word2Vec(cleaned_docs, min_count=5, vector_size=100, workers=4)

def vectorize_docs(docs, model):
    vectorized_docs = []
    for doc in docs:
        vectors = [model.wv[word] for word in doc if word in model.wv.key_to_index]
        if vectors:
            vectorized_docs.append(np.mean(vectors, axis=0))
        else:
            vectorized_docs.append(np.zeros(model.vector_size))
    return np.array(vectorized_docs)

# Vectorize your documents
vectorized_docs = vectorize_docs(cleaned_docs, w2v_model)

# Define the pipeline with SVM classifier
pipeline = Pipeline([
    ('classifier', SVC(kernel='rbf', C=130, probability=True))
])

# Splitting the Data for Training and Testing
labels = [doc['discipline'] for doc in descriptions]
X_train, X_test, y_train, y_test = train_test_split(vectorized_docs, labels, test_size=0.2, random_state=42)

# Fitting the vectorizer on the training data
pipeline.fit(X_train, y_train)

# Save the pipeline with an incremented count
existing_models = glob("C:/Users/LENOVO/Desktop/afcmflask/AFCM/venv/model/AFCM_pipeline*.joblib")
latest_counts = [int(re.search(r'\d+', model).group()) for model in existing_models if re.search(r'\d+', model) is not None]
latest_count = max(latest_counts, default=0) + 1
pipeline_output_path = f"C:/Users/LENOVO/Desktop/afcmflask/AFCM/venv/model/AFCM_pipeline{latest_count}.joblib"
joblib.dump(pipeline, pipeline_output_path)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)  # For ROC-AUC

# Evaluate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# For multi-class classification, we need to binarize the labels for ROC-AUC
y_test_bin = label_binarize(y_test, classes=pipeline.classes_)
y_pred_proba_bin = pipeline.predict_proba(X_test)

roc_auc = roc_auc_score(y_test_bin, y_pred_proba_bin, average='weighted', multi_class='ovr')

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")