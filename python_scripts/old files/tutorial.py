import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
from nltk.corpus import stopwords
import json
import glob
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Defining functions for loading data, writing data, and removing stopwords and other unnecessary data

def load_data(file):
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)

def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def remove_stops(text, stops):
    text = re.sub(r"AC\/\d{1,4}\/\d{1,4}", "", text)
    words = text.split()
    final = []
    for word in words:
        if word not in stops:
            final.append(word)
    final = " ".join(final)
    final = final.translate(str.maketrans("", "", string.punctuation))
    final = "".join([i for i in final if not i.isdigit()])
    while "  " in final:
        final = final.replace("  ", " ")
    return (final)

def clean_docs(docs):
    stops = stopwords.words("english")
    months = load_data("C:/Users/LENOVO/Desktop/python_scripts/months.json")
    stops = stops+months
    final = []
    for doc in docs:
        clean_doc = remove_stops(doc, stops)
        final.append(clean_doc)
    return (final)

# Loading the Data (Specifically)

descriptions = load_data("C:/Users/LENOVO/Desktop/python_scripts/trc_dn.json")["descriptions"]
names = load_data("C:/Users/LENOVO/Desktop/python_scripts/trc_dn.json")["names"]

print (descriptions[0])
#print (names[0])

# Cleaning the Data

cleaned_docs = clean_docs(descriptions)
#print (cleaned_docs[0])

# Defining the Vectorizer and Vectorizing Data

vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=100,
    max_df=0.8,
    min_df=5,
    ngram_range=(1,3),
    stop_words="english",
)

vectors = vectorizer.fit_transform(cleaned_docs)

# Getting the Keywords

feature_names = vectorizer.get_feature_names_out()

dense = vectors.todense()
denselist = dense.tolist()

all_keywords = []

for description in denselist:
    x=0
    keywords = []
    for word in description:
        if word > 0:
            keywords.append(feature_names[x])
        x=x+1
    all_keywords.append(keywords)

print (all_keywords[0])

# Clustering the Data and Defining the K-Means Model

true_k = 5

model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)

model.fit(vectors)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

with open ("C:/Users/LENOVO/Desktop/python_scripts/results.txt", "w", encoding="utf-8") as f:
    for i in range(true_k):
        f.write(f"Cluster {i}")
        f.write("\n")
        for ind in order_centroids[i, :10]:
            f.write(' %s' % terms[ind],)
            f.write("\n")
        f.write("\n")
        f.write("\n")

# Plotting the Clusters

kmean_indices = model.fit_predict(vectors)

pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(vectors.toarray())

colors = ["r", "b", "c", "y", "m"]

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]

fig, ax = plt.subplots(figsize=(50, 50))

ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])

for i, txt in enumerate(names):
    ax.annotate(txt[0:5], (x_axis[i], y_axis[i]))
    
plt.savefig("trc.png")

