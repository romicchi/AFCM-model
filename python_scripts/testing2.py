import joblib
import re
import pdfplumber
import string
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec

# Load the entire pipeline with Doc2Vec + SVM model
pipeline = joblib.load("C:/Users/LENOVO/Desktop/python_scripts/AFCM_pipeline6.joblib")

# Load the Doc2Vec model
doc2vec_model = Doc2Vec.load("C:/Users/LENOVO/Desktop/afcmflask/AFCM/venv/model/doc2vec_model.joblib")

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

# Function to predict discipline for an unlabeled PDF
def predict_discipline(pdf_path, model):
    with pdfplumber.open(pdf_path) as pdf:
        pdf_content = " ".join(page.extract_text() for page in pdf.pages)

    cleaned_text = remove_stops(pdf_content)

    # Tokenize the cleaned text
    tokenized_text = word_tokenize(cleaned_text)

    # Infer the vector using the Doc2Vec model
    vectorized_text = model.infer_vector(tokenized_text)

    # Reshape the vector to match the expected input for prediction
    vectorized_text = vectorized_text.reshape(1, -1)

    # Replace 'YourSVMModel' with the actual SVM model from your pipeline
    predicted_discipline = pipeline.predict(vectorized_text)

    return predicted_discipline[0]

# Example usage:
pdf_path = "C:/Users/LENOVO/Desktop/python_scripts/new resource/how language works.pdf"
predicted_discipline = predict_discipline(pdf_path, doc2vec_model)
print(f"Predicted Discipline: {predicted_discipline}")