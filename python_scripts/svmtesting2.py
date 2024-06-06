import joblib
import re
import pdfplumber
import string

# Load the pipeline
pipeline = joblib.load("C:/Users/LENOVO/Desktop/afcmflask/AFCM/venv/model/AFCM_pipeline9.joblib")

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

    # Remove extra whitespaces
    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()  # Remove leading and trailing whitespaces

def clean_docs(docs):
    # If docs is a string, directly process it
    if isinstance(docs, str):
        return remove_stops(docs)
    
    # If docs is a list of dictionaries, process each document
    final = []
    for doc in docs:
        clean_doc = remove_stops(doc['text'])
        final.append(clean_doc)
    return final

# Function to predict discipline for an unlabeled PDF
def predict_discipline(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        pdf_content = " ".join(page.extract_text() for page in pdf.pages)

    cleaned_text = clean_docs(pdf_content)

    # Ensure cleaned_text is a list of strings
    if isinstance(cleaned_text, str):
        cleaned_text = [cleaned_text]

    # Predict discipline
    predicted_discipline = pipeline.predict(cleaned_text)

    return predicted_discipline[0]

# Example usage:
pdf_path = "C:/Users/LENOVO/Desktop/python_scripts/new resource/exact definition of mathematics.pdf"
predicted_discipline = predict_discipline(pdf_path)
print(f"Predicted Discipline: {predicted_discipline}")