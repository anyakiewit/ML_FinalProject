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


import sys
import os
import json

### PARAMETERS ###
n_context_size = 10
context_output_path = "output/context_windows.jsonl"

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

    # Create context windows
    if os.path.exists(context_output_path):
        print("Loading cached context windows...")
        context_windows = load_context_windows_from_file(context_output_path)
    else:
        print("Generating and saving context windows...")
        context_windows = get_context_windows_padded(train_data, n_context_size)
        write_context_windows_to_file(context_windows, output_path=context_output_path)

    # Test statistical features on first real example
    if len(context_windows) > 0:
        first_example = context_windows[0]
        print("First context window:", first_example)

        context = " ".join(first_example["words"])
        target_word = first_example["target"]

        features = extract_statistical_features(context, target_word)
        print("Statistical features:", features)

    return 0

if __name__ == "__main__":
    sys.exit(main())
