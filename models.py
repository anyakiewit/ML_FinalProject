<<<<<<< Updated upstream
import gzip
import json
import os

import joblib
from rich import print
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from helper_functions import extract_statistical_features

# ---------------------- Helper Functions for Models ----------------------


def build_statistical_feature_matrix(context_windows):
    X = []
    y = []

    for example in context_windows:
        context = " ".join(example["words"])
        target_word = example["target"]
        label = example["target_label"]

        features = extract_statistical_features(
            context,
            target_word,
            example.get("word_index", 0),
            example.get("doc_length", 1)
        )
        X.append(features)
        y.append(label)

    return X, y


def build_combined_feature_matrix(
        context_windows,
        cache_path="output/train_mlm_features_cache.jsonl.gz"
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

    for example in context_windows:
        clean_context = " ".join([w for w in example['words'] if w != '<PAD>'])
        target_word = example["target"]
        label = example["target_label"]

        stat_features = extract_statistical_features(
            clean_context,
            target_word,
            example.get("word_index", 0),
            example.get("doc_length", 1)
        )

        window_key = f"{example['id']}_[{example['target']}]_{'_'.join(example['words'])}"

        mlm_features = mlm_cache.get(window_key, [0.0, 50000.0])
        combined = mlm_features + stat_features

        X.append(combined)
        y.append(label)

    return X, y


# ---------------------- Logistic Regression ----------------------


def train_logistic_regression(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, verbose=True))
    ])
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, split_name="Validation"):
    predictions = model.predict(X)
    print(f"\n[bold cyan]### {split_name} results ###[/bold cyan]")
    print(f"[bold green]Accuracy:[/bold green] {accuracy_score(y, predictions):.4f}")
    print(f"[dim]{classification_report(y, predictions, target_names=['Human', 'Machine'])}[/dim]")

# ---------------------- SVM ----------------------


# Standard SVM
def train_svm(
        X_train,
        y_train,
        kernel='linear',
        model_path="output/svm_model.joblib"
    ):

    if os.path.exists(model_path):
        print(f"[dim]Loading cached SVM model from {model_path}[/dim]")
        return joblib.load(model_path)

    print("[dim]Training new SVM model[/dim]")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel=kernel, class_weight='balanced', verbose=True))
    ])
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[dim]SVM model saved to {model_path}[/dim]")
    return model


# Linear SVM
def train_linear_svm(
        X_train,
        y_train,
        model_path="output/svm_linear_model.joblib"
    ):

    if os.path.exists(model_path):
        print(f"[dim]Loading cached SVM model from {model_path}[/dim]")
        return joblib.load(model_path)
    print("[dim]Training new SVM model[/dim]")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(class_weight='balanced', max_iter=2000))
    ])
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[dim]SVM model saved to {model_path}[/dim]")
    return model


def train_sdg_svm(
        X_train,
        y_train,
        model_path="output/svm_sgd_model.joblib"
    ):

    if os.path.exists(model_path):
        print(f"[dim]Loading cached SVM model from {model_path}[/dim]")
        return joblib.load(model_path)
    print("[dim]Training new SVM model[/dim]")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(loss='hinge', class_weight='balanced', n_jobs=-1))
    ])
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[dim]SVM model saved to {model_path}[/dim]")
    return model
=======
import gzip
import json
import os
import math
import numpy as np

import joblib
from rich import print
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

from helper_functions import extract_statistical_features

# ---------------------- Helper Functions for Models ----------------------


def build_statistical_feature_matrix(context_windows):
    X = []
    y = []

    for example in context_windows:
        context = " ".join(example["words"])
        target_word = example["target"]
        label = example["target_label"]

        features = extract_statistical_features(
            context,
            target_word
        )
        X.append(features)
        y.append(label)

    return X, y


def build_combined_feature_matrix(
        context_windows,
        cache_path="output/train_mlm_features_cache.jsonl.gz"
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

    for example in context_windows:
        clean_context = " ".join([w for w in example['words'] if w != '<PAD>'])
        target_word = example["target"]
        label = example["target_label"]

        stat_features = extract_statistical_features(
            clean_context,
            target_word
        )

        window_key = f"{example['id']}_[{example['target']}]_{'_'.join(example['words'])}"

        raw_prob = mlm_cache.get(window_key, [0.0, 50000.0])[0]
        mlm_surprisal = -math.log(max(raw_prob, 1e-10))
        combined = [mlm_surprisal] + stat_features

        X.append(combined)
        y.append(label)

    return X, y


# ---------------------- Logistic Regression ----------------------


def train_logistic_regression(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, verbose=True))
    ])
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, split_name="Validation"):
    predictions = model.predict(X)
    print(f"\n[bold cyan]### {split_name} results ###[/bold cyan]")
    print(f"[bold green]Accuracy:[/bold green] {accuracy_score(y, predictions):.4f}")
    print(f"[dim]{classification_report(y, predictions, target_names=['Human', 'Machine'])}[/dim]")

# ---------------------- SVM ----------------------


# Standard SVM
def train_svm(
        X_train,
        y_train,
        kernel='linear',
        model_path="output/svm_model.joblib"
    ):

    if os.path.exists(model_path):
        print(f"[dim]Loading cached SVM model from {model_path}[/dim]")
        return joblib.load(model_path)

    print("[dim]Training new SVM model[/dim]")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel=kernel, class_weight='balanced', verbose=True))
    ])
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[dim]SVM model saved to {model_path}[/dim]")
    return model


# Linear SVM
def train_linear_svm(
        X_train,
        y_train,
        model_path="output/svm_linear_model.joblib"
    ):

    if os.path.exists(model_path):
        print(f"[dim]Loading cached SVM model from {model_path}[/dim]")
        return joblib.load(model_path)
    print("[dim]Training new SVM model[/dim]")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(class_weight='balanced', max_iter=2000))
    ])
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[dim]SVM model saved to {model_path}[/dim]")
    return model


def train_sdg_svm(
        X_train,
        y_train,
        model_path="output/svm_sgd_model.joblib"
    ):

    if os.path.exists(model_path):
        print(f"[dim]Loading cached SVM model from {model_path}[/dim]")
        return joblib.load(model_path)
    print("[dim]Training new SVM model[/dim]")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(loss='hinge', class_weight='balanced', n_jobs=-1))
    ])
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[dim]SVM model saved to {model_path}[/dim]")
    return model


# ---------------------- Naive Bayes Baseline ----------------------


def train_naive_bayes_baseline(
        X_train_pos,
        y_train,
        model_path="output/nb_baseline_model.joblib"):
    """Trains a Naive Bayes baseline using ONLY the normalized document position."""

    if os.path.exists(model_path):
        print(f"[dim]Loading cached NB Baseline from {model_path}[/dim]")
        return joblib.load(model_path)

    print("[dim]Training new Naive Bayes Positional Baseline[/dim]")

    model = GaussianNB()
    model.fit(X_train_pos, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[dim]NB Baseline model saved to {model_path}[/dim]")
    return model

# ---------------------- Naive Bayes ----------------------

def train_naive_bayes(
        X_train,
        y_train,
        model_path="output/nb_full_model.joblib"):
    """Trains a Naive Bayes model using ALL provided features"""

    if os.path.exists(model_path):
        print(f"[dim]Loading cached full NB model from {model_path}[/dim]")
        return joblib.load(model_path)

    print("[dim]Training new full Naive Bayes model[/dim]")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB())
    ])
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[dim]full NB model saved to {model_path}[/dim]")
    return model
>>>>>>> Stashed changes
