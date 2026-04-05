from mlm_features import MLMFeatureExtractor
from pre_process import lemmatize_tokens
from pre_process import stem_tokens
from pre_process import load_data
from pre_process import create_validation_split_path
from context_window import get_context_windows_padded
from context_window import write_context_windows_to_file
from context_window import load_context_windows_from_file
# from sklearn.model_selection import train_test_split
from helper_functions import setup_nltk_data
from helper_functions import save_cached_data
from helper_functions import load_cached_data
from helper_functions import extract_statistical_features

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import sys
import os
# import json

### PARAMETERS ###
n_context_size = 10
context_output_path = "output/context_windows.jsonl"
dev_cache_path = "data/dev.jsonl"

BATCH_SIZE = 64

TEST_LIMIT = 50000

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

def main():
    """Main entry point of the application."""

    print("________________ Pre Processing ________________")

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

    lemmatizer = WordNetLemmatizer()

    print("________________ Context Windows ________________")

    # Create context windows for train
    if os.path.exists(context_output_path):
        print("Loading cached context windows...")
        train_context_windows = load_context_windows_from_file(context_output_path)
    else:
        print("Generating and saving context windows...")
        train_context_windows = get_context_windows_padded(train_data, n_context_size)
        write_context_windows_to_file(train_context_windows, output_path=context_output_path)

    # print("________________ Statistical Features ________________")
    # Create context windows for validation
    val_context_windows = get_context_windows_padded(val_data, n_context_size)

    # Optional: inspect first example
    if len(train_context_windows) > 0:
        first_example = train_context_windows[0]
        print("First context window:", first_example)

    # # Test statistical features on first real example
    # if len(context_windows) > 0:
    #     first_example = context_windows[0]
    #     print("First context window:", first_example)

    #     context = " ".join(first_example["words"])
    #     target_word = first_example["target"]

    #     features = extract_statistical_features(
    #         context, 
    #         target_word, 
    #         first_example.get("word_index", 0), 
    #         first_example.get("doc_length", 1))
    #     print("Statistical features:", features)

    print("________________ Perplexity Feature ________________")
    cache_path = "output/mlm_features_cache.jsonl"

    mlm_cache = {}
    
    if os.path.exists(cache_path):
        print(f"Loading cached MLM features from {cache_path}...")
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                mlm_cache[record["key"]] = [record["prob"], record["rank"], record.get("top_guess", "")]

    missing_windows = False
    for window in context_windows:
        window_key = f"{window['id']}_[{window['target']}]_{'_'.join(window['words'])}"
        if window_key not in mlm_cache:
            missing_windows = True
            break

    if missing_windows:
        mlm_extractor = MLMFeatureExtractor()
        with open(cache_path, 'a', encoding='utf-8') as f:
            for i in range(0, len(context_windows), BATCH_SIZE):
                batch = context_windows[i:i + BATCH_SIZE]
                batch_context_words, batch_target_words, batch_keys, uncached_indices = [], [], [], []

                for j, window in enumerate(batch):
                    window_key = f"{window['id']}_[{window['target']}]_{'_'.join(window['words'])}"
                    batch_keys.append(window_key)
                    if window_key not in mlm_cache:
                        batch_context_words.append(window['words'])
                        batch_target_words.append(window['target'])
                        uncached_indices.append(j)

                if uncached_indices:
                    batch_results = mlm_extractor.get_prediction_features_batch(batch_context_words, batch_target_words)
                    for idx, result in zip(uncached_indices, batch_results):
                        mlm_cache[batch_keys[idx]] = result
                        f.write(json.dumps({"key": batch_keys[idx], "prob": result[0], "rank": result[1], "top_guess": result[2]}) + '\n')
                
                if i % 1000 == 0: print(f"Processed {i}/{len(context_windows)} windows.")

    human_probs, human_ranks, machine_probs, machine_ranks = [], [], [], []

    final_dataset = []

    print("\n--- EXAMPLES OF MODEL PREDICTIONS ---")
    for idx, window in enumerate(context_windows):
        window_key = f"{window['id']}_[{window['target']}]_{'_'.join(window['words'])}"
        prob, rank, top_guess = mlm_cache[window_key]
        
        lemma = lemmatizer.lemmatize(window['target'].lower().strip(string.punctuation))

        clean_context = " ".join([w for w in window['words'] if w != '<PAD>'])

        stat_features = extract_statistical_features(
            context=clean_context,
            target_word=window['target'],
            word_index=window.get("word_index", 0),
            doc_length=window.get("doc_length", 1)
        )

        combined_features = {
            "id": window["id"],
            "target": window["target"],
            "lemma": lemma,
            "label": window["target_label"],
            "mlm_prob": prob,
            "mlm_rank": rank,
            "stat_avg_word_len": stat_features[0],
            "stat_sentence_len": stat_features[1],
            "stat_punct_count": stat_features[2],
            "stat_is_all_caps": stat_features[3],
            "stat_relative_pos": stat_features[4]
        }
        
        final_dataset.append(combined_features)

        if window["target_label"] == 0:
            human_probs.append(prob)
            human_ranks.append(rank)
        else:
            machine_probs.append(prob)
            machine_ranks.append(rank)

        if idx < 5:
            print(f"Context: {clean_context}")
            print(f"  Target: '{window['target']}' | Lemma: '{lemma}' | Rank: {rank} | Guess: '{top_guess}'")
            print(f"  Stats: {stat_features}")
            print("-" * 40)

    def print_stats(name, data):
        data_arr = np.array(data)
        print(f"  {name:12} -> Mean: {data_arr.mean():.6e} | Median: {np.median(data_arr):.6e}")

    print("\n" + "="*50)
    print(" MLM PREDICTION FEATURES ANALYSIS ")
    print("="*50)
    print(f"[ HUMAN ] Samples: {len(human_probs)}")
    print_stats("Probability", human_probs)
    print_stats("Rank", human_ranks)
    print(f"\n[ MACHINE ] Samples: {len(machine_probs)}")
    print_stats("Probability", machine_probs)
    print_stats("Rank", machine_ranks)
    print("="*50)

    # Build feature matrices
    X_train, y_train = build_statistical_feature_matrix(train_context_windows)
    X_val, y_val = build_statistical_feature_matrix(val_context_windows)

    # Train Logistic Regression
    model = train_logistic_regression(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_train, y_train, split_name="Train")
    evaluate_model(model, X_val, y_val, split_name="Validation")

    return 0

if __name__ == "__main__":
    sys.exit(main())
