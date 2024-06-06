import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import string
import json
import os
import re
import numpy as np

def load_data(folder):
    all_data = []

    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r', encoding="utf-8") as f:
                data = json.load(f)
                discipline = re.search(r'^(.*?)_', filename).group(1)
                data['discipline'] = discipline
                all_data.append(data)

    return all_data

def remove_stops(text):
    text = re.sub(r'M\d+_GADD\d+_\d+_SE_C01\.QXD \d+/\d+/\d+ \d+:\d+ [APMapm]{2} Page \d+', '', text)
    text = text.replace("\n", "")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join([i for i in text if not i.isdigit()])
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()

def clean_docs(docs):
    final = []
    for doc in docs:
        clean_doc = remove_stops(doc['text'])
        final.append(clean_doc)
    return final

# Load Data
descriptions = load_data("C:/Users/LENOVO/Desktop/python_scripts/json_resources")

# Cleaning the Data
cleaned_docs = clean_docs(descriptions)

# Splitting the Data for Training and Testing
labels = [doc['discipline'] for doc in descriptions]
X_train, X_test, y_train, y_test = train_test_split(cleaned_docs, labels, test_size=0.2, random_state=42)

# Define the pipeline without specific hyperparameters
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(lowercase=True,
                                   max_features=100,
                                   max_df=0.8,
                                   min_df=5,
                                   ngram_range=(1, 3),
                                   stop_words="english",
                                   n_jobs=-1  # Set n_jobs to -1 for parallelization
                                   )),
    ('classifier', SVC(kernel='linear', C=100, probability=True))
])

# Grid Search for Hyperparameter Tuning
param_grid = {
    'vectorizer__max_features': [100, 500, 1000],
    'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'classifier__C': [1, 10, 100],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

# Access the best model from the grid search
best_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred_grid = best_model.predict(X_test)

# Evaluate metrics
accuracy_grid = accuracy_score(y_test, y_pred_grid)
precision_grid = precision_score(y_test, y_pred_grid, average='weighted', zero_division=1)
recall_grid = recall_score(y_test, y_pred_grid, average='weighted')
f1_grid = f1_score(y_test, y_pred_grid, average='weighted')

print("Model Evaluation Metrics (Grid Search):")
print(f"Accuracy: {accuracy_grid}")
print(f"Precision: {precision_grid}")
print(f"Recall: {recall_grid}")
print(f"F1 Score: {f1_grid}")
