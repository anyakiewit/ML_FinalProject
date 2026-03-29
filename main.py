from mlm_features import MLMFeatureExtractor
from pre_process import lemmatize_tokens
from pre_process import stem_tokens
from pre_process import load_data
from pre_process import create_validation_split_path
from context_window import get_context_windows_padded
from context_window import write_context_windows_to_file
from context_window import load_context_windows_from_file
from sklearn.model_selection import train_test_split
from helper_functions import setup_nltk_data
from helper_functions import save_cached_data
from helper_functions import load_cached_data
from helper_functions import extract_statistical_features

import numpy as np

import sys
import os
import json

### PARAMETERS ###
n_context_size = 10
context_output_path = "output/context_windows.jsonl"

def main():
    """Main entry point of the application."""

    # ________________ Pre Processing ________________

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

    # ________________ Context Windows ________________

    # Create context windows
    if os.path.exists(context_output_path):
        print("Loading cached context windows...")
        context_windows = load_context_windows_from_file(context_output_path)
    else:
        print("Generating and saving context windows...")
        context_windows = get_context_windows_padded(train_data, n_context_size)
        write_context_windows_to_file(context_windows, output_path=context_output_path)

    # ________________ Statistical Features ________________

    # Test statistical features on first real example
    if len(context_windows) > 0:
        first_example = context_windows[0]
        print("First context window:", first_example)

        context = " ".join(first_example["words"])
        target_word = first_example["target"]

        features = extract_statistical_features(context, target_word)
        print("Statistical features:", features)

    # ________________ Perplexity Feature ________________

    mlm_extractor = None

    cache_path = "output/mlm_features_cache.json"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"Loading cached MLM features from {cache_path}...")
        with open(cache_path, 'r', encoding='utf-8') as f:
            mlm_cache = json.load(f)
    else:
        mlm_cache = {}

    human_probs, human_ranks = [], []
    machine_probs, machine_ranks = [], []

    windows_to_process = context_windows

    for i, window in enumerate(windows_to_process):
        if i % 100 == 0 and i > 0:
            print(f"Processed {i} windows")

        target_word = window["target"]
        context_words = window["words"]
        label = window["target_label"]

        context_str = "_".join(context_words)
        window_key = f"{window['id']}_[{target_word}]_{context_str}"

        if window_key in mlm_cache:
            prob, rank = mlm_cache[window_key]
        else:
            if mlm_extractor is None:
                mlm_extractor = MLMFeatureExtractor()

            prob, rank = mlm_extractor.get_prediction_features(context_words, target_word)

            mlm_cache[window_key] = [prob, rank]

            if len(mlm_cache) % 100 == 0:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(mlm_cache, f)

        if label == 0:
            human_probs.append(prob)
            human_ranks.append(rank)
        else:
            machine_probs.append(prob)
            machine_ranks.append(rank)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(mlm_cache, f)
        print(f"Successfully saved {len(mlm_cache)} features to cache.")

    def print_stats(name, data):
        if not data:
            print(f"{name}: No data available")
            return
        data_arr = np.array(data)
        print(f"  {name:12} -> Mean: {data_arr.mean():.6f} | Median: {np.median(data_arr):.6f} | Min: {data_arr.min():.6f} | Max: {data_arr.max():.6f}")

    print("\n" + "="*50)
    print(" MLM PREDICTION FEATURES ANALYSIS")
    print("="*50)

    print("Mean: This score is the model's confidence (from 0.0 to 1.0) that the actual word is the correct answer")
    print("")
    print("Rank: A rank of 1 means it was the model's top guess. A rank of 40,000 means the model thought it was not a valid guess.")

    print("")
    print("[ HUMAN WRITTEN WORDS ]")
    print(f"  Total samples: {len(human_probs)}")
    print_stats("Probability", human_probs)
    print_stats("Rank", human_ranks)

    print("")
    print("[ MACHINE GENERATED WORDS ]")
    print(f"  Total samples: {len(machine_probs)}")
    print_stats("Probability", machine_probs)
    print_stats("Rank", machine_ranks)
    print("")

    print("="*50)

    return 0

if __name__ == "__main__":
    sys.exit(main())
