
import gzip
import json
import os
import string
import sys

import numpy as np
from nltk import WordNetLemmatizer
from rich import print

from context_window import (get_context_windows_padded,
                            load_context_windows_from_file,
                            write_context_windows_to_file)
from helper_functions import (extract_statistical_features, load_cached_data,
                              save_cached_data, setup_nltk_data)
from mlm_features import (MLMFeatureExtractor, analyze_mlm_predictions,
                          get_or_create_mlm_features)
from models import (build_combined_feature_matrix,
                    build_statistical_feature_matrix, evaluate_model,
                    train_linear_svm, train_logistic_regression, train_sdg_svm,
                    train_svm, train_naive_bayes_baseline)
from pre_process import create_validation_split_path, load_data
from visualize import plot_feature_importances

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
    nb_model = train_naive_bayes_baseline(X_train, y_train)

    X_val_pos = np.array(X_val)[:, -1].reshape(-1, 1)
    evaluate_model(nb_model, X_val_pos, y_val, split_name="Validation Baseline")

    print("\n[bold magenta]________________ Logistic Regression ________________[/bold magenta]")

    # Train Logistic Regression
    model = train_logistic_regression(X_train, y_train)
    model_comb = train_logistic_regression(X_train_comb, y_train_comb)

    # Evaluate
    evaluate_model(model, X_train, y_train, split_name="Train")
    evaluate_model(model, X_val, y_val, split_name="Validation")
    evaluate_model(model_comb, X_train_comb, y_train_comb, split_name="Train Combined")
    evaluate_model(model_comb, X_val_comb, y_val_comb, split_name="Validation Combined")

    print("\n[bold magenta]________________ SVM ________________[/bold magenta]")

    # Build feature matrices
    X_train, y_train = build_statistical_feature_matrix(train_context_windows)
    X_val, y_val = build_statistical_feature_matrix(val_context_windows)

    X_train_comb, y_train_comb = build_combined_feature_matrix(train_context_windows, cache_path=TRAIN_CACHE_PATH)
    X_val_comb, y_val_comb = build_combined_feature_matrix(val_context_windows, cache_path=VAL_CACHE_PATH)

    ### Trial of SVM variants ###

    # Train SVM on Statistical Features
    svm_linear_model = train_linear_svm(X_train, y_train)
    svm_sgd_model = train_sdg_svm(X_train, y_train)

    # Train SVM on Combined Features
    svm_linear_model_comb = train_linear_svm(X_train_comb, y_train_comb, model_path="output/svm_linear_model_comb.joblib")
    svm_sgd_model_comb = train_sdg_svm(X_train_comb, y_train_comb, model_path="output/svm_sgd_model_comb.joblib")

    # Evaluate
    print("\n[magenta]________________ SVM Linear ________________[/magenta]")
    evaluate_model(svm_linear_model, X_train, y_train, split_name="Train")
    evaluate_model(svm_linear_model, X_val, y_val, split_name="Validation")

    print("\n[magenta]________________ SVM SGD ________________[/ magenta]")
    evaluate_model(svm_sgd_model, X_train, y_train, split_name="Train")
    evaluate_model(svm_sgd_model, X_val, y_val, split_name="Validation")

    print("\n[magenta]________________ SVM Linear Combined ________________[/magenta]")
    evaluate_model(svm_linear_model_comb, X_train_comb, y_train_comb, split_name="Train Combined")
    evaluate_model(svm_linear_model_comb, X_val_comb, y_val_comb, split_name="Validation Combined")

    print("\n[magenta]________________ SVM SGD Combined ________________[/magenta]")
    evaluate_model(svm_sgd_model_comb, X_train_comb, y_train_comb, split_name="Train Combined")
    evaluate_model(svm_sgd_model_comb, X_val_comb, y_val_comb, split_name="Validation Combined")


    print("\n[bold magenta]________________ Feature Importances ________________[/bold magenta]")

    stat_feature_names = [
        "word_length", "is_capitalized", "is_numeric", 
        "has_punctuation", "doc_length", "punctuation_count", "doc_pos", "norm_doc_pos"
    ]
    comb_feature_names = ["mlm_probability", "mlm_rank"] + stat_feature_names

    plot_feature_importances(model, feature_names=stat_feature_names, title="Logistic Regression feature importance")
    plot_feature_importances(model_comb, feature_names=comb_feature_names, title="Logistic Regression Combined feature importance")

    plot_feature_importances(svm_linear_model, feature_names=stat_feature_names, title="Linear SVM feature importance")
    plot_feature_importances(svm_sgd_model, feature_names=stat_feature_names, title="SGD SVM feature importance")

    plot_feature_importances(svm_linear_model_comb, feature_names=comb_feature_names, title="Linear SVM Combined feature importance")
    plot_feature_importances(svm_sgd_model_comb, feature_names=comb_feature_names, title="SGD SVM Combined feature importance")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
