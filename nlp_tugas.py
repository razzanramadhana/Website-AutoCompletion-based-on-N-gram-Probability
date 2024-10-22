from flask import Flask, request, jsonify, render_template
import pandas as pd
from collections import defaultdict, Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data
df = pd.read_csv("/Users/razzanramadhana/Documents/SEMESTER 5/NLP/data.csv")
df = df.dropna(subset=['title', 'url'])
titles = df['title'].tolist()
urls = df['url'].tolist()

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(titles)

# Create a dictionary for title-url mapping
title_url_dict = dict(zip(titles, urls))

# Initialize bigram model
n_grams = defaultdict(Counter)
for title in titles:
    words = re.findall(r'\w+', title.lower())
    for i in range(len(words) - 1):
        n_grams[words[i]][words[i + 1]] += 1

def search_tfidf(query):
    """Search function based on TF-IDF and converts cosine similarity to probability in decimal format."""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    total_similarity = sum(similarities)  # Sum all similarities for normalization

    results = []
    for idx in similarities.argsort()[-5:][::-1]:  # Get the indices of the top 5 results
        title = titles[idx]
        url = title_url_dict.get(title)
        probability = (similarities[idx] / total_similarity) if total_similarity > 0 else 0  # Normalize to probability
        results.append({"title": title, "url": url, "probability": round(probability, 4)})  # Format as decimal
    return results

def autocomplete_ngram(prefix):
    """Autocomplete function using n-grams with refined probability calculation."""
    words = re.findall(r'\w+', prefix.lower())
    
    if not words:
        return []  # If the input is empty, no n-grams can be used

    last_word = words[-1]
    total_bigrams = sum(n_grams[last_word].values())  # Total number of bigrams starting with the last word
    results = []

    # Using both bigrams and trigrams to find relevant titles
    for title in titles:
        title_words = re.findall(r'\w+', title.lower())
        matches = 0
        possible_matches = 0

        # Count bigram matches between the query and the title
        for i in range(len(title_words) - 1):
            if title_words[i] == last_word:
                possible_matches += 1
                if i < len(title_words) - 1 and title_words[i + 1] in n_grams[last_word]:
                    matches += n_grams[last_word][title_words[i + 1]]

        # Calculate probability based on matches
        probability = (matches / total_bigrams) if total_bigrams > 0 else 0
        results.append({"title": title, "url": title_url_dict.get(title), "probability": round(probability, 4)})

    # Sort results by probability and limit to top 5
    results = sorted(results, key=lambda x: x['probability'], reverse=True)[:5]

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search_api():
    query = request.args.get('query', '')
    method = request.args.get('method', 'tfidf')  # 'tfidf' or 'ngram'
    if method == 'tfidf':
        suggestions = search_tfidf(query)
    elif method == 'ngram':
        suggestions = autocomplete_ngram(query)
    else:
        suggestions = []
    return jsonify(suggestions)

if __name__ == "__main__":
    app.run(debug=True)
