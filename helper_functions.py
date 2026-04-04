import nltk
import os
import json
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def setup_nltk_data():
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    print("### Downloading NLTK data ###")

    nltk.download('wordnet', download_dir=nltk_data_dir)
    nltk.download('omw-1.4', download_dir=nltk_data_dir)

    print("### NLTK data downloaded ###")


def save_cached_data(data, output_path):
    """Save processed data to a JSONL cache file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for record in data:
            json.dump(record, outfile)
            outfile.write('\n')

def load_cached_data(input_path):
    """Load processed data from a JSONL cache file."""
    records = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            records.append(json.loads(line))
    return records

def average_word_length(context):
    words = [word for word in context.split() if word != "<PAD>"]
    if len(words) == 0:
        return 0.0
    return sum(len(word) for word in words) / len(words)

def sentence_length(context):
    words = [word for word in context.split() if word != "<PAD>"]
    return len(words)

def punctuation_count(context):
    words = [word for word in context.split() if word != "<PAD>"]
    clean_context = " ".join(words)
    return sum(1 for char in clean_context if char in string.punctuation)

def all_caps_function(target_word):
    return 1 if target_word.isupper() and any(c.isalpha() for c in target_word) else 0

def extract_statistical_features(context, target_word):
    return [
        average_word_length(context),
        sentence_length(context),
        punctuation_count(context),
        all_caps_function(target_word)
    ]

def build_tfidf_vectorizer(context_windows):
    """Fit a TF-IDF vectorizer on training context window texts (lowercased)."""
    corpus = [
        " ".join(w for w in cw["words"] if w != "<PAD>").lower()
        for cw in context_windows
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    return vectorizer

def extract_tfidf_score(vectorizer, context_words, target_word):
    """Get the TF-IDF score of the target word within its context window."""
    context_text = " ".join(w for w in context_words if w != "<PAD>").lower()
    target_lower = target_word.lower()

    tfidf_matrix = vectorizer.transform([context_text])
    vocab = vectorizer.vocabulary_

    if target_lower in vocab:
        col = vocab[target_lower]
        return float(tfidf_matrix[0, col])
    return 0.0