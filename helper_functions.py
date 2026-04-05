import json
import os
import string

import nltk


def setup_nltk_data():
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    print("### Downloading NLTK data ###")

    nltk.download('wordnet', download_dir=nltk_data_dir)
    nltk.download('omw-1.4', download_dir=nltk_data_dir)

    print("### NLTK data downloaded ###")


def save_cached_data(data, output_path):
    """Save processed data to a JSONL cache file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for record in data:
            json.dump(record, outfile)
            outfile.write('\n')


def load_cached_data(input_path):
    """Load processed data from a JSONL cache file."""
    records = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            records.append(json.loads(line))
    return records


def average_word_length(context):
    words = [word for word in context.split() if word != "<PAD>"]
    if len(words) == 0:
        return 0.0
    return sum(len(word) for word in words) / len(words)


# TODO: do not use this function as it very hard to implement correctly.
def sentence_length(context):
    words = [word for word in context.split() if word != "<PAD>"]
    return len(words)


def punctuation_count(context):
    words = [word for word in context.split() if word != "<PAD>"]
    clean_context = " ".join(words)
    return sum(1 for char in clean_context if char in string.punctuation)


def all_caps_function(target_word):
    return 1 if target_word.isupper() and any(c.isalpha() for c in target_word) else 0


def extract_statistical_features(context, target_word, word_index, doc_length):
    relative_position = word_index / doc_length if doc_length > 0 else 0.0

    return [
        average_word_length(context),
        punctuation_count(context),
        all_caps_function(target_word),
        relative_position
    ]
