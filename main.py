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
from models import (evaluate_model,train_linear_svm, train_logistic_regression, train_sdg_svm, train_naive_bayes_baseline, train_naive_bayes, train_random_forest, evaluate_results, calculate_boundary_mae)
from pre_process import create_validation_split_path, load_data
from visualize import plot_feature_importances
from stat_features import build_combined_feature_matrix, build_statistical_feature_matrix, build_positional_feature_matrix, build_tfidf_vectorizer
from sklearn.metrics import accuracy_score, f1_score

# PARAMETERS
N_CONTEXT_SIZE = 10
BATCH_SIZE = 512
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
            windows = get_context_windows_padded(data, N_CONTEXT_SIZE)
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
    analyze_mlm_predictions(train_context_windows, train_mlm_cache, lemmatizer, show_examples=False, split_name="Train")

    print("\n[magenta]________________ Validation ________________[/magenta]")
    analyze_mlm_predictions(val_context_windows, val_mlm_cache, lemmatizer, show_examples=False, split_name="Validation")


    print("\n[magenta]________________ Feature Matrices Preparation ________________[/magenta]")
    
    # TF-IDF Setup
    print("[dim]Fitting TF-IDF Vectorizer on Train[/dim]")
    tfidf_vectorizer = build_tfidf_vectorizer(train_context_windows)

    print("\n[magenta]________________ Building Feature Matrices ________________[/magenta]")

    # Build feature matrices
    # TODO speed this up maybe?
    X_train, y_train = build_statistical_feature_matrix(train_context_windows, vectorizer=tfidf_vectorizer)
    X_val, y_val = build_statistical_feature_matrix(val_context_windows, vectorizer=tfidf_vectorizer)

    X_train_comb, y_train_comb = build_combined_feature_matrix(train_context_windows, cache_path=TRAIN_CACHE_PATH, vectorizer=tfidf_vectorizer)
    X_val_comb, y_val_comb = build_combined_feature_matrix(val_context_windows, cache_path=VAL_CACHE_PATH, vectorizer=tfidf_vectorizer)

    # list for all the results
    results_list = []

    print("\n[magenta]________________ NB Baseline ________________[/magenta]")
    X_train_pos, y_train_pos = build_positional_feature_matrix(train_context_windows)
    X_val_pos, y_val_pos = build_positional_feature_matrix(val_context_windows)

    nb_model = train_naive_bayes_baseline(X_train_pos, y_train_pos)

    evaluate_model(nb_model, X_val_pos, y_val_pos, split_name="Validation Baseline")

    # Add to table data
    preds_base = nb_model.predict(X_val_pos)
    results_list.append({
        'name': "Baseline NB",
        'accuracy': accuracy_score(y_val_pos, preds_base),
        'f1': f1_score(y_val_pos, preds_base, average='macro'),
        'mae': calculate_boundary_mae(preds_base, val_context_windows)
    })

    print("\n[bold magenta]________________ Full Naive Bayes ________________[/bold magenta]")

    # Train Full Naive Bayes
    nb_full = train_naive_bayes(X_train, y_train)
    nb_full_comb = train_naive_bayes(X_train_comb, y_train_comb, model_path="output/nb_full_model_comb.joblib")

    # Evaluate
    evaluate_model(nb_full, X_train, y_train, context_windows=train_context_windows, split_name="Train")
    evaluate_model(nb_full, X_val, y_val, context_windows=val_context_windows, split_name="Validation")
    evaluate_model(nb_full_comb, X_train_comb, y_train_comb, context_windows=train_context_windows, split_name="Train Combined")
    evaluate_model(nb_full_comb, X_val_comb, y_val_comb, context_windows=val_context_windows, split_name="Validation Combined")

    # Add to table data
    preds_fullNB = nb_full_comb.predict(X_val_comb)
    results_list.append({
        'name': "Full NB (Combined)",
        'accuracy': accuracy_score(y_val_comb, preds_fullNB),
        'f1': f1_score(y_val_comb, preds_fullNB, average='macro'),
        'mae': calculate_boundary_mae(preds_fullNB, val_context_windows)
    })

    print("\n[bold magenta]________________ Logistic Regression ________________[/bold magenta]")

    # Train Logistic Regression
    model = train_logistic_regression(X_train, y_train)
    model_comb = train_logistic_regression(X_train_comb, y_train_comb)

    # Evaluate
    evaluate_model(model, X_train, y_train, context_windows=train_context_windows, split_name="Train")
    evaluate_model(model, X_val, y_val, context_windows=val_context_windows, split_name="Validation")
    evaluate_model(model_comb, X_train_comb, y_train_comb, context_windows=train_context_windows, split_name="Train Combined")
    evaluate_model(model_comb, X_val_comb, y_val_comb, context_windows=val_context_windows, split_name="Validation Combined")

    # Add to table data
    preds_lr = model_comb.predict(X_val_comb)
    results_list.append({
        'name': "Logistic Regression (Combined)",
        'accuracy': accuracy_score(y_val_comb, preds_lr),
        'f1': f1_score(y_val_comb, preds_lr, average='macro'),
        'mae': calculate_boundary_mae(preds_lr, val_context_windows)
    })

    print("\n[bold magenta]________________ SVM ________________[/bold magenta]")

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

    # add to table data
    preds_svm_lin = svm_linear_model.predict(X_val)
    results_list.append({
        'name': "SVM Linear (Standard) - Val",
        'accuracy': accuracy_score(y_val, preds_svm_lin),
        'f1': f1_score(y_val, preds_svm_lin, average='macro'),
        'mae': calculate_boundary_mae(preds_svm_lin, val_context_windows)
    })

    print("\n[magenta]________________ SVM SGD ________________[/ magenta]")
    evaluate_model(svm_sgd_model, X_train, y_train, context_windows=train_context_windows, split_name="Train")
    evaluate_model(svm_sgd_model, X_val, y_val, context_windows=val_context_windows, split_name="Validation")

    # add to table data
    preds_svm_sgd = svm_sgd_model.predict(X_val)
    results_list.append({
        'name': "SVM SGD (Standard) - Val",
        'accuracy': accuracy_score(y_val, preds_svm_sgd),
        'f1': f1_score(y_val, preds_svm_sgd, average='macro'),
        'mae': calculate_boundary_mae(preds_svm_sgd, val_context_windows)
    })

    print("\n[magenta]________________ SVM Linear Combined ________________[/magenta]")
    evaluate_model(svm_linear_model_comb, X_train_comb, y_train_comb, context_windows=train_context_windows, split_name="Train Combined")
    evaluate_model(svm_linear_model_comb, X_val_comb, y_val_comb, context_windows=val_context_windows, split_name="Validation Combined")

    # add to table data
    preds_svm_lin_c = svm_linear_model_comb.predict(X_val_comb)
    results_list.append({
        'name': "SVM Linear (Combined) - Val",
        'accuracy': accuracy_score(y_val_comb, preds_svm_lin_c),
        'f1': f1_score(y_val_comb, preds_svm_lin_c, average='macro'),
        'mae': calculate_boundary_mae(preds_svm_lin_c, val_context_windows)
    })

    print("\n[magenta]________________ SVM SGD Combined ________________[/magenta]")
    evaluate_model(svm_sgd_model_comb, X_train_comb, y_train_comb, context_windows=train_context_windows, split_name="Train Combined")
    evaluate_model(svm_sgd_model_comb, X_val_comb, y_val_comb, context_windows=val_context_windows, split_name="Validation Combined")

    # add to table data
    preds_svm_sgd_c = svm_sgd_model_comb.predict(X_val_comb)
    results_list.append({
        'name': "SVM SGD (Combined) - Val",
        'accuracy': accuracy_score(y_val_comb, preds_svm_sgd_c),
        'f1': f1_score(y_val_comb, preds_svm_sgd_c, average='macro'),
        'mae': calculate_boundary_mae(preds_svm_sgd_c, val_context_windows)
    })

    print("\n[bold magenta]________________ Random Forest ________________[/bold magenta]")

    rf_model = train_random_forest(X_train, y_train, tune=True)
    rf_model_comb = train_random_forest(X_train_comb, y_train_comb, model_path="output/rf_model_comb.joblib", tune=True)

    evaluate_model(rf_model, X_train, y_train, context_windows=train_context_windows, split_name="Train")
    evaluate_model(rf_model, X_val, y_val, context_windows=val_context_windows, split_name="Validation")
    evaluate_model(rf_model_comb, X_train_comb, y_train_comb, context_windows=train_context_windows, split_name="Train Combined")
    evaluate_model(rf_model_comb, X_val_comb, y_val_comb, context_windows=val_context_windows, split_name="Validation Combined")

    # add to table data
    preds_rf = rf_model_comb.predict(X_val_comb)
    results_list.append({
        'name': "RF Combined - Validation",
        'accuracy': accuracy_score(y_val_comb, preds_rf),
        'f1': f1_score(y_val_comb, preds_rf, average='macro'),
        'mae': calculate_boundary_mae(preds_rf, val_context_windows)
    })

    print("\n[bold magenta]________________ Feature Importances ________________[/bold magenta]")

    stat_feature_names = [
        "average_word_length", 
        "punctuation_count", 
        "is_all_caps", 
        "is_first_capitalized",
        "tfidf_score"
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

    plot_feature_importances(rf_model, feature_names=stat_feature_names, title="Random Forest feature importance")
    plot_feature_importances(rf_model_comb, feature_names=comb_feature_names, title="Random Forest Combined feature importance")

    print("\n[bold magenta]________________ Final Evaluation ________________[/bold magenta]")

    evaluate_results(results_list)


    print("\n[bold green]________________ FINAL TEST PHASE ________________[/bold green]")

    # 1. Load test data
    test_data = load_data("raw_data/subtaskC_dev.jsonl")

    # 2. Extract test context windows and MLM features
    TEST_CONTEXT_PATH = "output/test_context_windows.jsonl.gz"
    TEST_MLM_CACHE_PATH = "output/test_mlm_features_cache.jsonl.gz"

    test_windows = get_or_create_context_windows(test_data, TEST_CONTEXT_PATH)
    test_mlm_cache = get_or_create_mlm_features(test_windows, TEST_MLM_CACHE_PATH, batch_size=BATCH_SIZE)

    # 3. Build test feature matrix (Combined)
    X_test_comb, y_test_comb = build_combined_feature_matrix(
        test_windows, 
        cache_path=TEST_MLM_CACHE_PATH, 
        vectorizer=tfidf_vectorizer
    )

    # 4. Test evaluate already trained Random Forest
    print("\n[magenta]________________ Random Forest Results on Test Data ________________[/magenta]")
    evaluate_model(
        rf_model_comb, 
        X_test_comb, 
        y_test_comb, 
        context_windows=test_windows,
        split_name="FINAL TEST SET"
    )

    preds_test = rf_model_comb.predict(X_test_comb)

    results_list.append({
        'name': "Random Forest Combined - Test",
        'accuracy': accuracy_score(y_test_comb, preds_test),
        'f1': f1_score(y_test_comb, preds_test, average='macro'),
        'mae': calculate_boundary_mae(preds_test, test_windows)
    })

    return 0

if __name__ == "__main__":
    sys.exit(main())
