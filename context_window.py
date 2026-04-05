import gzip
import json
import os


def get_context_windows_padded(data, n_context):
    """Create context windows from record having n_context to either side"""
    """Stores target word with label and words with their labels"""
    """Identifiable by id, originating from original record"""

    context_windows = []

    pad_word = '<PAD>'
    pad_label = -1

    for record in data:
        id = record['id']
        words = record['words']
        labels = record['labels']

        padded_words = [pad_word] * n_context + words + [pad_word] * n_context
        padded_labels = [pad_label] * n_context + labels + [pad_label] * n_context

        length = len(words)

        for i in range(length):
            start = i
            end = i + (2 * n_context) + 1

            context_windows.append({
                'id': id,
                'target': words[i],
                'target_label': labels[i],
                'words': padded_words[start:end],
                'labels': padded_labels[start:end],
                'word_index': i,
                'doc_length': length
            })

    return context_windows


def write_context_windows_to_file(windows, output_path):
    """Write context windows to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    open_func = gzip.open if output_path.endswith('.gz') else open
    with open_func(output_path, 'wt', encoding='utf-8') as outfile:
        for window in windows:
            json.dump(window, outfile)
            outfile.write('\n')


def load_context_windows_from_file(input_path):
    """Load context windows from an existing file"""
    context_windows = []
    open_func = gzip.open if input_path.endswith('.gz') else open
    with open_func(input_path, 'rt', encoding='utf-8') as infile:
        for line in infile:
            context_windows.append(json.loads(line))
    return context_windows
