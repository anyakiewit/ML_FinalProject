from pre_process import lemmatize_tokens
from pre_process import stem_tokens
from pre_process import load_data
from pre_process import create_validation_split_path

from context_window import get_context_windows_padded
from context_window import write_context_windows_to_file
from context_window import load_context_windows_from_file

from helper_functions import setup_nltk_data
from helper_functions import save_cached_data
from helper_functions import load_cached_data
from helper_functions import extract_statistical_features
from helper_functions import build_tfidf_vectorizer
from helper_functions import extract_tfidf_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split

import sys
import os
# import json

### PARAMETERS ###
n_context_size = 10
context_output_path = "output/context_windows.jsonl"

def build_statistical_feature_matrix(context_windows):
    X = []
    y = []

    for example in context_windows:
        context = " ".join(example["words"])
        target_word = example["target"]
        label = example["target_label"]

        features = extract_statistical_features(context, target_word)
        X.append(features)
        y.append(label)

    return X, y

def train_logistic_regression(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, split_name="Validation"):
    predictions = model.predict(X)
    print(f"\n### {split_name} results ###")
    print("Accuracy:", accuracy_score(y, predictions))
    print(classification_report(y, predictions))

def build_statistical_feature_matrix(context_windows, vectorizer=None):
    X = []
    y = []

    for example in context_windows:
        context = " ".join(example["words"])
        target_word = example["target"]
        label = example["target_label"]

        features = extract_statistical_features(context, target_word)

        if vectorizer is not None:
            tfidf_score = extract_tfidf_score(vectorizer, example["words"], target_word)
            features.append(tfidf_score)

        X.append(features)
        y.append(label)

    return X, y

def main():
    """Main entry point of the application."""

    dev_cache_path = "data/dev.jsonl"

    # Create validation split
    create_validation_split_path("raw_data/subtaskC_train.jsonl")

    # Load data
    train_data = load_data("data/train.jsonl")
    val_data = load_data("data/val.jsonl")

    # Load dev data -- TESTING DATA DO NOT USE UNTIL TESTING
    if os.path.exists(dev_cache_path):
        dev_data = load_cached_data(dev_cache_path)
    else:
        dev_data = load_data("raw_data/subtaskC_dev.jsonl")
        save_cached_data(dev_data, dev_cache_path)

    # Download wordnet and omw-1.4; Choice for models is ungrounded
    setup_nltk_data()

    # Lemmatize tokens
    for record in train_data + val_data + dev_data:
        record['words'] = lemmatize_tokens(record['words'])

    # Stem tokens
    # for record in train_data + val_data + dev_data:
    #     record['words'] = stem_tokens(record['words'])

    # Create context windows for train
    if os.path.exists(context_output_path):
        print("Loading cached context windows...")
        train_context_windows = load_context_windows_from_file(context_output_path)
    else:
        print("Generating and saving context windows...")
        train_context_windows = get_context_windows_padded(train_data, n_context_size)
        write_context_windows_to_file(train_context_windows, output_path=context_output_path)

    # Create context windows for validation
    val_context_windows = get_context_windows_padded(val_data, n_context_size)
    
    # Create TF-IDF vectorizer
    tfidf_vectorizer = build_tfidf_vectorizer(train_context_windows)

    # Optional: inspect first example
    if len(train_context_windows) > 0:
        first_example = train_context_windows[0]
        print("First context window:", first_example)

        context = " ".join(first_example["words"])
        target_word = first_example["target"]

        features = extract_statistical_features(context, target_word)
        print("Statistical features:", features)

        tfidf_score = extract_tfidf_score(tfidf_vectorizer, first_example["words"], target_word)
        print(f"TF-IDF score for target word '{target_word}': {tfidf_score:.6f}")

    # Build feature matrices
    X_train, y_train = build_statistical_feature_matrix(train_context_windows, tfidf_vectorizer)
    X_val, y_val = build_statistical_feature_matrix(val_context_windows, tfidf_vectorizer)

    # Train Logistic Regression
    model = train_logistic_regression(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_train, y_train, split_name="Train")
    evaluate_model(model, X_val, y_val, split_name="Validation")

    return 0

if __name__ == "__main__":
    sys.exit(main())
