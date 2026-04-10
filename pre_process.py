import json
import os

from sklearn.model_selection import train_test_split

def load_data(input_path):
    """Load data from a JSONL file."""

    output_records = []
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Required data file not found: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            record = json.loads(line)

            words = record['text'].split(" ")
            boundary = record['label']

            labels = []
            for i in range(len(words)):
                if i < boundary:
                    labels.append(0)
                else:
                    labels.append(1)

            output_record = {
                'id': record['id'],
                'words': words,
                'labels': labels
            }

            output_records.append(output_record)

    return output_records


def create_validation_split_path(train_data_path, test_size=0.2, random_state=42):
    """Create a validation split from the training data."""

    train_out_path = 'data/train.jsonl'
    val_out_path = 'data/val.jsonl'

    if os.path.exists(train_out_path) and os.path.exists(val_out_path):
        return

    os.makedirs('data', exist_ok=True)

    with open(train_data_path, 'r', encoding='utf-8') as infile:
        records = infile.readlines()

    train_data, val_data = train_test_split(records, test_size=test_size, random_state=random_state)

    with open(train_out_path, 'w', encoding='utf-8') as outfile:
        for record in train_data:
            outfile.write(record)

    with open(val_out_path, 'w', encoding='utf-8') as outfile:
        for record in val_data:
            outfile.write(record)

    return train_data, val_data
