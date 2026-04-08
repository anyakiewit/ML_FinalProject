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
import gzip
import json
import os
import string
import sys

import numpy as np
from nltk import WordNetLemmatizer
from rich import print

from context_window import (get_context_windows_padded,load_context_windows_from_file,write_context_windows_to_file)
from helper_functions import (load_cached_data,save_cached_data, setup_nltk_data,)
from mlm_features import (analyze_mlm_predictions, get_or_create_mlm_features)
from models import (evaluate_model,train_linear_svm, train_logistic_regression, train_sdg_svm, train_naive_bayes_baseline, train_naive_bayes)
from pre_process import create_validation_split_path, load_data
from visualize import plot_feature_importances
from stat_features import build_combined_feature_matrix, build_statistical_feature_matrix, build_positional_feature_matrix

# PARAMETERS
N_CONTEXT_SIZE = 10
BATCH_SIZE = 64
TEST_LIMIT = 50000

# ENV VARIABLES
TRAIN_CONTEXT_OUTPUT_PATH = "output/train_context_windows.jsonl.gz"
VAL_CONTEXT_OUTPUT_PATH = "output/val_context_windows.jsonl.gz"

TRAIN_CACHE_PATH = "output/train_mlm_features_cache.jsonl.gz"
VAL_CACHE_PATH = "output/val_mlm_features_cache.jsonl.gz"

DEV_CACHE_PATH = "data/dev.jsonl"


def main():
    """Main entry point of the application."""

    print("\n[bold magenta]________________ Pre Processing ________________[/bold magenta]")

    # Create validation split
    create_validation_split_path("raw_data/subtaskC_train.jsonl")

    # Load data
    train_data = load_data("data/train.jsonl")
    val_data = load_data("data/val.jsonl")

    # Load dev data -- TESTING DATA DO NOT USE UNTIL TESTING
    if os.path.exists(DEV_CACHE_PATH):
        dev_data = load_cached_data(DEV_CACHE_PATH)
    else:
        dev_data = load_data("raw_data/subtaskC_dev.jsonl")
        save_cached_data(dev_data, DEV_CACHE_PATH)

    # Download wordnet and omw-1.4; Choice for models is ungrounded
    setup_nltk_data()

    lemmatizer = WordNetLemmatizer()

    print("\n[bold magenta]________________ Context Windows ________________[/bold magenta]")

    def get_or_create_context_windows(data, output_path):
        if os.path.exists(output_path):
            print(f"[dim]Loading cached context windows from {output_path}[/dim]")
            return load_context_windows_from_file(output_path)
        else:
            print(f"[dim]Generating and saving context windows to {output_path}[/dim]")
            windows = get_context_windows_padded(data, n_context_size)
            write_context_windows_to_file(windows, output_path=output_path)
            return windows

    train_context_windows = get_or_create_context_windows(train_data, TRAIN_CONTEXT_OUTPUT_PATH)
    val_context_windows = get_or_create_context_windows(val_data, VAL_CONTEXT_OUTPUT_PATH)

    if len(train_context_windows) > 0:
        print(f"\n[dim]First train context window: {train_context_windows[0]}[/dim]")

    print("\n[bold magenta]________________ MLM Features ________________[/bold magenta]")

    train_mlm_cache = get_or_create_mlm_features(train_context_windows, TRAIN_CACHE_PATH, batch_size=BATCH_SIZE)
    val_mlm_cache = get_or_create_mlm_features(val_context_windows, VAL_CACHE_PATH, batch_size=BATCH_SIZE)

    if len(train_mlm_cache) > 0:
        first_key, first_val = next(iter(train_mlm_cache.items()))
        print(f"\n[dim]First train MLM cache entry:\n  Key: {first_key}\n  Value: {first_val}[/dim]")

    print("\n[magenta]________________ Train ________________[/magenta]")
    analyze_mlm_predictions(train_context_windows, train_mlm_cache, lemmatizer, show_examples=False)

    print("\n[magenta]________________ Validation ________________[/magenta]")
    analyze_mlm_predictions(val_context_windows, val_mlm_cache, lemmatizer, show_examples=False)


    # Build feature matrices
    X_train, y_train = build_statistical_feature_matrix(train_context_windows)
    X_val, y_val = build_statistical_feature_matrix(val_context_windows)

    X_train_comb, y_train_comb = build_combined_feature_matrix(train_context_windows, cache_path=TRAIN_CACHE_PATH)
    X_val_comb, y_val_comb = build_combined_feature_matrix(val_context_windows, cache_path=VAL_CACHE_PATH)

    print("\n[magenta]________________ NB Baseline (Position Only) ________________[/magenta]")
    X_train_pos, y_train_pos = build_positional_feature_matrix(train_context_windows)
    X_val_pos, y_val_pos = build_positional_feature_matrix(val_context_windows)

    nb_model = train_naive_bayes_baseline(X_train_pos, y_train_pos)

    evaluate_model(nb_model, X_val_pos, y_val_pos, split_name="Validation Baseline")

    print("\n[bold magenta]________________ Full Naive Bayes ________________[/bold magenta]")

    # Train Full Naive Bayes
    nb_full = train_naive_bayes(X_train, y_train)
    nb_full_comb = train_naive_bayes(X_train_comb, y_train_comb, model_path="output/nb_full_model_comb.joblib")

    # Evaluate
    evaluate_model(nb_full, X_train, y_train, context_windows=train_context_windows, split_name="Train")
    evaluate_model(nb_full, X_val, y_val, context_windows=val_context_windows, split_name="Validation")
    evaluate_model(nb_full_comb, X_train_comb, y_train_comb, context_windows=train_context_windows, split_name="Train Combined")
    evaluate_model(nb_full_comb, X_val_comb, y_val_comb, context_windows=val_context_windows, split_name="Validation Combined")

    print("\n[bold magenta]________________ Logistic Regression ________________[/bold magenta]")

    # Train Logistic Regression
    model = train_logistic_regression(X_train, y_train)
    model_comb = train_logistic_regression(X_train_comb, y_train_comb)

    # Evaluate
    evaluate_model(model, X_train, y_train, context_windows=train_context_windows, split_name="Train")
    evaluate_model(model, X_val, y_val, context_windows=val_context_windows, split_name="Validation")
    evaluate_model(model_comb, X_train_comb, y_train_comb, context_windows=train_context_windows, split_name="Train Combined")
    evaluate_model(model_comb, X_val_comb, y_val_comb, context_windows=val_context_windows, split_name="Validation Combined")

    print("\n[bold magenta]________________ SVM ________________[/bold magenta]")

    # Build feature matrices
    X_train, y_train = build_statistical_feature_matrix(train_context_windows)
    X_val, y_val = build_statistical_feature_matrix(val_context_windows)

    X_train_comb, y_train_comb = build_combined_feature_matrix(train_context_windows, cache_path=TRAIN_CACHE_PATH)
    X_val_comb, y_val_comb = build_combined_feature_matrix(val_context_windows, cache_path=VAL_CACHE_PATH)

    ### Trial of SVM variants ###

    # Train SVM on Statistical Features; Linear has hyperparameter tuning
    svm_linear_model = train_linear_svm(X_train, y_train, tune=True)
    svm_sgd_model = train_sdg_svm(X_train, y_train)

    # Train SVM on Combined Features
    svm_linear_model_comb = train_linear_svm(X_train_comb, y_train_comb, model_path="output/svm_linear_model_comb.joblib", tune=True)
    svm_sgd_model_comb = train_sdg_svm(X_train_comb, y_train_comb, model_path="output/svm_sgd_model_comb.joblib")

    # Evaluate
    print("\n[magenta]________________ SVM Linear ________________[/magenta]")
    evaluate_model(svm_linear_model, X_train, y_train, context_windows=train_context_windows, split_name="Train")
    evaluate_model(svm_linear_model, X_val, y_val, context_windows=val_context_windows, split_name="Validation")

    print("\n[magenta]________________ SVM SGD ________________[/ magenta]")
    evaluate_model(svm_sgd_model, X_train, y_train, context_windows=train_context_windows, split_name="Train")
    evaluate_model(svm_sgd_model, X_val, y_val, context_windows=val_context_windows, split_name="Validation")

    print("\n[magenta]________________ SVM Linear Combined ________________[/magenta]")
    evaluate_model(svm_linear_model_comb, X_train_comb, y_train_comb, context_windows=train_context_windows, split_name="Train Combined")
    evaluate_model(svm_linear_model_comb, X_val_comb, y_val_comb, context_windows=val_context_windows, split_name="Validation Combined")

    print("\n[magenta]________________ SVM SGD Combined ________________[/magenta]")
    evaluate_model(svm_sgd_model_comb, X_train_comb, y_train_comb, context_windows=train_context_windows, split_name="Train Combined")
    evaluate_model(svm_sgd_model_comb, X_val_comb, y_val_comb, context_windows=val_context_windows, split_name="Validation Combined")


    print("\n[bold magenta]________________ Feature Importances ________________[/bold magenta]")

    stat_feature_names = [
        "average_word_length", 
        "punctuation_count", 
        "is_all_caps", 
        "is_first_capitalized"
    ]
    comb_feature_names = [
        "mlm_log_prob", 
        "mlm_log_rank", 
        "mlm_perplexity"
    ] + stat_feature_names

    plot_feature_importances(model, feature_names=stat_feature_names, title="Logistic Regression feature importance")
    plot_feature_importances(model_comb, feature_names=comb_feature_names, title="Logistic Regression Combined feature importance")

    plot_feature_importances(svm_linear_model, feature_names=stat_feature_names, title="Linear SVM feature importance")
    plot_feature_importances(svm_sgd_model, feature_names=stat_feature_names, title="SGD SVM feature importance")

    plot_feature_importances(svm_linear_model_comb, feature_names=comb_feature_names, title="Linear SVM Combined feature importance")
    plot_feature_importances(svm_sgd_model_comb, feature_names=comb_feature_names, title="SGD SVM Combined feature importance")
    
    plot_feature_importances(nb_model, feature_names=["doc_pos", "norm_doc_pos"], title="Baseline Naive Bayes feature importance")
    plot_feature_importances(nb_full, feature_names=stat_feature_names, title="Full Naive Bayes feature importance")
    plot_feature_importances(nb_full_comb, feature_names=comb_feature_names, title="Full Naive Bayes Combined feature importance")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
