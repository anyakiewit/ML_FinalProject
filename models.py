from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
import gzip
import json
import os
import math
import numpy as np

import joblib
from rich import print
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# ---------------------- Logistic Regression ----------------------


def train_logistic_regression(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, verbose=True))
    ])
    model.fit(X_train, y_train)
    return model


def calculate_boundary_mae(predictions, context_windows):
    doc_boundaries = {}
    
    for idx, window in enumerate(context_windows):
        doc_id = window['id']
        if doc_id not in doc_boundaries:
            doc_boundaries[doc_id] = {'actual': -1, 'predicted': -1, 'length': window['doc_length']}
        
        if window['target_label'] == 1 and doc_boundaries[doc_id]['actual'] == -1:
            doc_boundaries[doc_id]['actual'] = window['word_index']
            
        if predictions[idx] == 1 and doc_boundaries[doc_id]['predicted'] == -1:
            doc_boundaries[doc_id]['predicted'] = window['word_index']

    mae_errors = []
    for doc_id, bounds in doc_boundaries.items():
        actual = bounds['actual'] if bounds['actual'] != -1 else bounds['length']
        predicted = bounds['predicted'] if bounds['predicted'] != -1 else bounds['length']
        
        mae_errors.append(abs(actual - predicted))
    
    if mae_errors:
        mae = np.mean(mae_errors)
        print(f"[bold yellow]Boundary MAE (in word index offset):[/bold yellow] {mae:.2f}")
        return mae
    return None


def evaluate_model(model, X, y, context_windows=None, split_name="Validation"):
    predictions = model.predict(X)

    macro_f1 = f1_score(y, predictions, average='macro')

    print(f"\n[bold cyan]### {split_name} results ###[/bold cyan]")
    print(f"[bold green]Accuracy:[/bold green] {accuracy_score(y, predictions):.4f}")
    print(f"[bold green]Macro F1:[/bold green] {macro_f1:.4f}")
    print(f"[dim]{classification_report(y, predictions, target_names=['Human', 'Machine'])}[/dim]")

    if context_windows is not None:
        calculate_boundary_mae(predictions, context_windows)

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
        model_path="output/svm_linear_model.joblib",
        tune=True
    ):

    if os.path.exists(model_path):
        print(f"[dim]Loading cached SVM model from {model_path}[/dim]")
        return joblib.load(model_path)

    print("[dim]Training new SVM model[/dim]")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(class_weight='balanced', max_iter=3000, dual=False)) 
    ])
    if tune:
        print("[dim]Running RandomizedSearchCV for Hyperparameters[/dim]")
        param_dist = {'clf__C': loguniform(1e-4, 1e2)}
        search = RandomizedSearchCV(pipeline, param_dist, n_iter=5, cv=3, scoring='f1_macro', n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        print(f"[bold green]Best Linear SVM Params:[/bold green] {search.best_params_}")
        model = search.best_estimator_
    else:
        model = pipeline.fit(X_train, y_train)

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
