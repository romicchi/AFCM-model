import fitz  # PyMuPDF
import langid
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tkinter import Tk, filedialog
import re

nltk.download('stopwords')

# Function to process text in chunks
def process_text_chunks(text, chunk_size=100000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    processed_text = ''
    stop_words = set(stopwords.words('english'))
    for chunk in chunks:
        tokens = nltk.word_tokenize(chunk)
        filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
        processed_text += ' '.join(filtered_tokens)
    return processed_text

# Step 1: PDF Parsing
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

# Step 2: Language Detection with LangID
def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

# Step 3: Disciplines

# Updated Disciplines
computer_science_disciplines = [
    "algorithm", "data structures", "programming", "software development", 
    "artificial intelligence", "machine learning", "computer vision", "code", 
    "robotics", "cybersecurity", "database", "cloud computing", "operating systems", 
    "network security", "web development", "mobile app development", "big data", 
    "cryptography", "programming languages", "internet-of-things", "computer architecture", 
    "compiler", "software engineering", "human-computer-interaction", "parallel computing", 
    "database management", "game development", "data mining"
]

philosophy_disciplines = [
    "philosophy", "metaphysics", "epistemology", "ethics", "logic", "aesthetics", 
    "existentialism", "socrates", "plato", "aristotle", "free will", "idealism", 
    "syllogism", "absurdism", "utilitarianism", "phenomenology", "skepticism", 
    "ontology", "virtue", "empiricism", "rationalism", "nihilism", "phenomenalism", 
    "solipsism", "materialism", "dualism", "monism", "objectivism"
]

psychology_disciplines = [
    "psychology", "behavior", "mind", "cognition", "emotion", "mental health", 
    "therapy", "clinical psychology", "counseling psychology", "developmental psychology", 
    "social psychology", "cognitive psychology", "biopsychology", "abnormal psychology", 
    "neuroscience", "personality", "stress", "memory", "perception", "motivation", 
    "cognitive dissonance", "operant conditioning", "observational learning", 
    "locus of control", "social cognition", "self-efficacy", "self-actualization", 
    "attachment theory"
]

social_sciences_disciplines = [
    "sociology", "psychology", "anthropology", "economics", "political science", 
    "geography", "history", "cultural studies", "social psychology", "linguistics", 
    "criminology", "social work", "education", "demography", "archaeology", 
    "communication studies", "international relations", "urban planning", 
    "human development", "gender studies", "public policy", "social welfare", 
    "environmental studies", "media studies", "social research", "ethnography", 
    "social justice", "social policy"
]

mathematics_disciplines = [
    "algebra", "geometry", "calculus", "statistics", "arithmetic", "trigonometry", 
    "linear algebra", "differential equations", "number theory", "probability", 
    "combinatorics", "topology", "abstract algebra", "set theory", "mathematical analysis", 
    "vector calculus", "complex analysis", "linear programming", "discrete mathematics", 
    "cryptography", "numerical analysis", "multivariable calculus", "group theory", 
    "real analysis", "graph theory", "geometry of surfaces", "mathematical logic", 
    "partial differential equations"
]

natural_sciences_disciplines = [
    "biology", "chemistry", "physics", "astronomy", "geology", "earth science", 
    "ecology", "zoology", "botany", "environmental science", "paleontology", 
    "meteorology", "oceanography", "microbiology", "genetics", "biochemistry", 
    "quantum mechanics", "thermodynamics", "nuclear physics", "organic chemistry", 
    "inorganic chemistry", "physical chemistry", "astro physics", "cosmology", 
    "plate tectonics", "climatology", "volcanology", "seismology"
]

arts_disciplines = [
    "visual arts", "performing arts", "fine arts", "music", "dance", "painting", 
    "sculpture", "theater", "film", "photography", "drawing", "design", "architecture", 
    "literature", "poetry", "drama", "graphic design", "ceramics", "printmaking", 
    "animation", "digital art", "classical music", "contemporary art", "ballet", 
    "jazz", "opera", "creative writing", "art history"
]

sports_disciplines = [
    "athletics", "basketball", "soccer", "football", "baseball", "tennis", 
    "swimming", "volleyball", "golf", "track and field", "cycling", "gymnastics", 
    "wrestling", "martial arts", "cricket", "rugby", "hockey", "badminton", 
    "table tennis", "archery", "bowling", "rowing", "sailing", "skateboarding", 
    "surfing", "skiing", "snowboarding", "triathlon"
]

language_disciplines = [
    "language", "grammar", "syntax", "phonetics", "phonology", "morphology", 
    "semantics", "pragmatics", "language acquisition", "psycholinguistics", 
    "sociolinguistics", "dialect", "bilingualism", "multilingualism", 
    "language variation", "language change", "language revitalization", 
    "language documentation", "language preservation", "translation", 
    "interpretation", "language teaching", "second language acquisition", 
    "language assessment", "corpus linguistics", "discourse analysis", 
    "speech recognition", "natural language processing"
]

linguistics_disciplines = [
    "phonetics", "phonology", "morphology", "syntax", "semantics", "pragmatics", 
    "sociolinguistics", "psycholinguistics", "neurolinguistics", "corpus linguistics", 
    "descriptive linguistics", "prescriptive linguistics", "generative grammar", 
    "transformational grammar", "structural linguistics", "cognitive linguistics", 
    "functional linguistics", "discourse analysis", "semiotics", "language typology", 
    "constrastive linguistics", "dialectology", "sociophonetics", "lexicography", 
    "etymology", "linguistic anthropology", "linguistc pragmatics", 
    "phonological rules"
]

literature_disciplines = [
    "poetry", "prose", "fiction", "nonfiction", "drama", "novel", "short story", 
    "epic", "lyric", "sonnet", "haiku", "rhyme", "meter", "symbolism", "allegory", 
    "metaphor", "simile", "irony", "satire", "genre", "tragedy", "comedy", "plot", 
    "characterization", "setting", "theme", "motif", "foreshadowing"
]

geography_disciplines = [
    "geography", "physical geography", "human geography", "cartography", 
    "geospatial analysis", "geographic information systems", "topography", 
    "climate", "weather", "ecosystem", "biome", "plate tectonics", "landforms", 
    "urban geography", "rural geography", "population geography", "demographics", 
    "migration", "urbanization", "environmental geography", "cultural geography", 
    "regional geography", "geopolitics", "transportation geography", 
    "economic geography", "agriculture", "natural resources", "sustainability"
]

history_disciplines = [
    "history", "historiography", "historian", "primary source", "secondary source", 
    "archaeology", "paleography", "oral history", "ancient history", "medieval history", 
    "modern history", "contemporary history", "world history", "regional history", 
    "social history", "cultural history", "political history", "economic history", 
    "military history", "diplomatic history", "intellectual history", "gender history", 
    "labor history", "environmental history", "public history", "art history", 
    "history education", "historical methodology"
]

management_disciplines = [
    "management", "leadership", "organization", "business administration", 
    "strategic planning", "human resource management", "financial management", 
    "operations management", "marketing management", "project management", 
    "supply chain management", "quality management", "risk management", 
    "change management", "conflic management", "crisis management", "time management", 
    "decision making", "team management", "performance management", 
    "organizational culture", "entrepreneurship", "innovation", 
    "management consulting", "corporate governance", "stakeholder management", 
    "management information systems", "management theory"
]

# Step 5: User Interface
def get_file_path():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select a PDF file")
    return file_path


# Main Script
if __name__ == "__main__":
    # Step 5: User Interface
    pdf_path = get_file_path()

    # Step 1: PDF Parsing with PyMuPDF
    pdf_text = extract_text_from_pdf(pdf_path)

    # Step 2: Language Detection with LangID
    detected_language = detect_language(pdf_text)

    # Step 3: Process Text in Chunks with NLTK and Remove Stopwords and Special Characters
    processed_text = process_text_chunks(pdf_text)

    # Step 4: TF/IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([processed_text])

    # Step 5: Calculate TF/IDF Scores
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), vectors.toarray()[0]))

    # Step 6: Match Disciplines
    matched_disciplines = []
    
    # Use the appropriate discipline list for the context, for example, arts_disciplines
    for discipline in arts_disciplines:
        discipline_keywords = discipline.split()
        discipline_score = sum(tfidf_scores.get(key, 0) for key in discipline_keywords)
        matched_disciplines.append((discipline, discipline_score))
    
    # Step 7: Choose the most accurate discipline
    matched_disciplines.sort(key=lambda x: x[1], reverse=True)
    selected_discipline = matched_disciplines[0][0] if matched_disciplines else None
    
    # Display results
    document_name = pdf_path.split('\\')[-1]  # Extracting the document name from the file path
    print(f"Title: {document_name}")
    print(f"Detected Language: {detected_language}")
    print(f"Selected Discipline: {selected_discipline}")
