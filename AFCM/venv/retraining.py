import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import string
import json
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import TruncatedSVD
from glob import glob
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import requests
from datetime import datetime
import gdown
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define the folder_path containing the training data
folder_path = "C:/Users/LENOVO/Desktop/python_scripts/json_resources"

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

    # Perform stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Join the words back into a single string
    text = " ".join(words)

    # Remove extra whitespaces
    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()  # Remove leading and trailing whitespaces

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

def train_svm_model():
    
    # Load Data
    descriptions = load_data(folder_path)

    # Cleaning the Data
    cleaned_docs = clean_docs(descriptions)

    # Define the pipeline with TF-IDF vectorizer, TruncatedSVD and SVM classifier
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(lowercase=True,
                                       max_features=100,
                                       max_df=0.8,
                                       min_df=5,
                                       ngram_range=(1, 3),
                                       stop_words="english"
                                       )),
        ('svd', TruncatedSVD(n_components=50)),  # Use TruncatedSVD to reduce dimensionality
        ('classifier', SVC(kernel='rbf', C=130, probability=True))
    ])

    # Splitting the Data for Training and Testing
    labels = [doc['discipline'] for doc in descriptions]
    X_train, X_test, y_train, y_test = train_test_split(cleaned_docs, labels, test_size=0.2, random_state=42)

    # Fitting the vectorizer on the training data
    pipeline.fit(X_train, y_train)

    # Save the pipeline with an incremented count
    existing_models = glob("C:/Users/LENOVO/Desktop/python_scripts/AFCM_pipeline*.joblib")
    latest_counts = [int(re.search(r'\d+', model).group()) for model in existing_models if re.search(r'\d+', model) is not None]
    latest_count = max(latest_counts, default=0) + 1
    pipeline_output_path = f"C:/Users/LENOVO/Desktop/python_scripts/AFCM_pipeline{latest_count}.joblib"
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

def check_date():
    try:
        current_date = datetime.now()
        if current_date.day == 4:  # Changed this line
            # Send a GET request to the Laravel application
            response = requests.get('https://gener-lnulib.site/api/resources')

            # Check if the request was successful
            if response.status_code == 200:
                # Convert the response to JSON
                resources = response.json()

                # Find all existing JSON files
                existing_files = glob("C:/Users/LENOVO/Desktop/python_scripts/test_folder/resources*.json")

                # Extract the counts from the filenames
                counts = [int(re.search(r'\d+', file).group()) for file in existing_files if re.search(r'\d+', file) is not None]

                # Find the maximum count and increment it for the new file
                count = max(counts, default=0) + 1

                # Write the resources to a new JSON file with the incremented count
                with open(f'C:/Users/LENOVO/Desktop/python_scripts/test_folder/resources{count}.json', 'w') as f:
                    json.dump(resources, f)
                    
                resources_file = max(glob("C:/Users/LENOVO/Desktop/python_scripts/test_folder/resources*.json"))  # get the latest resources file

                with open(resources_file, 'r') as f:
                    if f.read().strip():
                        f.seek(0)  # Reset the file pointer to the beginning
                        resources = json.load(f)
                    else:
                        resources = []

                x = len(resources)  # Total number of resources
                y = 0  # Number of replaced files

                for resource in resources:
                    url = resource['json_url']
                    # Extract Google Drive ID from url
                    file_id = url.split('=')[-1]
                    gdrive_url = f'https://drive.google.com/uc?id={file_id}'
                    # Get the original filename from the Content-Disposition header
                    response = requests.head(gdrive_url, allow_redirects=True)
                    content_disposition = response.headers.get('content-disposition')
                    if content_disposition:
                        filename = re.findall('filename[^;=\n]*=([^;\n]*)', content_disposition)
                        if filename:
                            filename = filename[0].strip('"')
                        else:
                            filename = 'default_filename'  # Use a default filename if the original filename cannot be found
                    else:
                        filename = 'default_filename'  # Use a default filename if the Content-Disposition header is not present
                    output = os.path.join(folder_path, filename)
                    if os.path.exists(output):
                        y += 1  # Increment the count of replaced files
                    gdown.download(gdrive_url, output, quiet=False)
                
                    # Check if the downloaded file is empty
                    if os.path.getsize(output) == 0:
                        print(f"File {output} is empty. Deleting...")
                        os.remove(output)

                # Check if the number of replaced files is equal to the total number of resources
                if x == y:
                    print("Training skipped. No new resources have been uploaded.")
                else:
                    # Call the train_svm_model function
                    train_svm_model()
                    print("SVM model re-training completed successfully.")
            else:
                print("Failed to get resources from Laravel application.")
        else:
            print("SVM model training skipped. Today is not the 1st.")
    except Exception as e:
        print(f"Error during SVM model training: {e}")

# Train the SVM model
check_date()
