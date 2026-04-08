from sklearn.feature_extraction.text import TfidfVectorizer
import math
import gzip
import json
import string
import joblib
import os

def average_word_length(words):
    if len(words) == 0:
        return 0.0
    return sum(len(word) for word in words) / len(words)

# TODO: do not use this function as it very hard to implement correctly.
def sentence_length(context):
    return len(context)

def punctuation_count(words):
    clean_context = " ".join(words)
    return sum(1 for char in clean_context if char in string.punctuation)


def all_caps_function(target_word):
    return 1 if target_word.isupper() and any(c.isalpha() for c in target_word) else 0

def is_first_capitalized(target_word):
    "detects if the first character of the target word is capitalized"
    if not target_word:
        return 0
    if target_word[0].isupper():
        return 1
    else:
        return 0

def doc_position(word_index):
    "returns the absolute position of the word"
    return word_index

def normalized_doc_position(word_index, doc_length):
    "returns the relative position of the word (0.0 to 1.0)"
    if doc_length <= 0:
        return 0.0
    return word_index/doc_length

def extract_statistical_features(context, target_word):
    return [
        average_word_length(context),
        punctuation_count(context),
        all_caps_function(target_word),
        is_first_capitalized(target_word)
    ]

# ---------------------- TF-IDF ----------------------

def build_tfidf_vectorizer(context_windows, cache_path="output/tfidf_vectorizer.joblib"):
    """Fit a TF-IDF vectorizer on training context window texts (lowercased) or load cached one."""
    if os.path.exists(cache_path):
        from rich import print
        print(f"[dim]Loading cached TF-IDF vectorizer from {cache_path}[/dim]")
        return joblib.load(cache_path)

    corpus = [
        " ".join(w for w in cw["words"] if w != "<PAD>").lower()
        for cw in context_windows
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    joblib.dump(vectorizer, cache_path)
    
    from rich import print
    print(f"[dim]Saved TF-IDF vectorizer to {cache_path}[/dim]")
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


# ---------------------- Helper Functions for Models ----------------------


def build_statistical_feature_matrix(context_windows, vectorizer=None):
    X = []
    y = []

    for example in context_windows:
        clean_words = [w for w in example["words"] if w != "<PAD>"]
        target_word = example["target"]
        label = example["target_label"]

        features = extract_statistical_features(
            clean_words,
            target_word
        )
        
        if vectorizer is not None:
            tfidf_score = extract_tfidf_score(vectorizer, example["words"], target_word)
            features.append(tfidf_score)

        X.append(features)
        y.append(label)

    return X, y


def build_combined_feature_matrix(
        context_windows,
        cache_path="output/train_mlm_features_cache.jsonl.gz",
        vectorizer=None
    ):

    mlm_cache = {}
    open_func = gzip.open if cache_path.endswith('.gz') else open
    try:
        with open_func(cache_path, 'rt', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                mlm_cache[record["key"]] = [record["prob"], record["rank"]]
    except FileNotFoundError:
        print(f"[yellow]Warning: Cache file {cache_path} not found.[/yellow]")

    X = []
    y = []

    EPSILON = 1e-15

    for example in context_windows:
        clean_words = [w for w in example['words'] if w != '<PAD>']
        target_word = example["target"]
        label = example["target_label"]

        stat_features = extract_statistical_features(
            clean_words,
            target_word
        )
        
        if vectorizer is not None:
            tfidf_score = extract_tfidf_score(vectorizer, example["words"], target_word)
            stat_features.append(tfidf_score)

        window_key = f"{example['id']}_[{example['target']}]_{'_'.join(example['words'])}"

        raw_prob, raw_rank = mlm_cache.get(window_key, [EPSILON, 50000.0])

        safe_prob = max(raw_prob, EPSILON)
        
        log_prob = math.log(safe_prob)

        log_rank = math.log(max(raw_rank, 1.0))

        perplexity = min(1.0 / safe_prob, 1e5)

        engineered_mlm_features = [log_prob, log_rank, perplexity]

        combined = engineered_mlm_features + stat_features
        
        X.append(combined)
        y.append(label)

    return X, y

def build_positional_feature_matrix(context_windows):
    """Builds a matrix containing ONLY doc_pos and normalized_doc_pos"""
    X = []
    y = []
    for example in context_windows:
        word_index = example.get("word_index", 0)
        doc_length = example.get("doc_length", 1)
        normalized_pos = word_index / doc_length if doc_length > 0 else 0.0
        
        X.append([word_index, normalized_pos])
        y.append(example["target_label"])
        
    return X, y