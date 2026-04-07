import gzip
import json
import math
import os
import string

import numpy as np
import torch
from rich import print
from transformers import AutoModelForMaskedLM, AutoTokenizer

class MLMFeatureExtractor:
    def __init__(self, model_name="roberta-large"):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("[dim]Using GPU.[/dim]")
        else:
            self.device = torch.device("cpu")
            print("[dim]Using CPU.[/dim]")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    # Method currently not used, but kept for reference and maybe future use.
    def get_window_feature(self, context_words):

        # Remove <PAD> tokens, Model handles padding on its own.
        clean_words = []
        for word in context_words:
            if word != "<PAD>":
                clean_words.append(word)

        text = " ".join(clean_words)

        # Precaution if a context-window happens to be empty.
        if not text.strip():
            return 0.0 * self.model.config.hidden_size

        # Prepares data so model can understand it.
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # Running the model.
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extracts the "summary" feature vector. This vector represents the entire context window.
        cls_feature = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return cls_feature

    # Using a masked language model to predict the target word.
    def get_prediction_features_batch(
            self,
            batch_context_words,
            batch_target_words
        ):

        batch_masked_texts = []

        for context_words in batch_context_words:
            middle_idx = len(context_words) // 2
            masked_words = context_words.copy()
            masked_words[middle_idx] = self.tokenizer.mask_token

            clean_words = [w for w in masked_words if w != "<PAD>"]
            batch_masked_texts.append(" ".join(clean_words))

        inputs = self.tokenizer(
            batch_masked_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = []

        # Iterates through each example in the batch to extract the target word probability and rank.
        for i in range(len(batch_masked_texts)):
            mask_token_index = torch.where(inputs["input_ids"][i] == self.tokenizer.mask_token_id)[0]

            if len(mask_token_index) == 0:
                results.append([0.0, 50000.0, ""])
                continue

            mask_token_index = mask_token_index[0].item()
            mask_token_logits = outputs.logits[i, mask_token_index, :].clone()

            # Mask out special tokens so they aren't predicted.
            mask_token_logits[self.tokenizer.all_special_ids] = -float('inf')

            # Converts logits to a probability distribution.
            mask_token_probs = torch.softmax(mask_token_logits, dim=0)

            # As the model tokenizes words, the actual target word also needs to be tokenized.
            clean_target = batch_target_words[i].strip(string.punctuation)
            target_token_ids = self.tokenizer(" " + clean_target, add_special_tokens=False)["input_ids"]

            if not target_token_ids:
                results.append([0.0, 50000.0, ""])
                continue

            total_log_prob = 0.0
            best_rank = 50000.0

            # Iterates through each token in the target word to calculate the probability and rank.
            for token_id in target_token_ids:
                prob = mask_token_probs[token_id].item()

                total_log_prob += math.log(max(prob, 1e-10))

                rank = (mask_token_probs > prob).sum().item() + 1

                if rank < best_rank:
                    best_rank = rank

            # Aggregates the log probabilities and converts them to a single probability.
            avg_log_prob = total_log_prob / len(target_token_ids)
            target_prob = math.exp(avg_log_prob)

            top_token_id = torch.argmax(mask_token_probs).item()
            top_guess = self.tokenizer.decode([top_token_id]).strip()

            results.append([target_prob, float(best_rank), top_guess])

        return results

# ---------------------- MLM Helper Functions ----------------------


def get_or_create_mlm_features(
        context_windows,
        cache_path,
        batch_size=64
    ):

    mlm_cache = {}
    open_func = gzip.open if cache_path.endswith('.gz') else open

    if os.path.exists(cache_path):
        print(f"[dim]Loading cached MLM features from {cache_path}[/dim]")
        with open_func(cache_path, 'rt', encoding='utf-8') as f:
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
        with open_func(cache_path, 'at', encoding='utf-8') as f:
            for i in range(0, len(context_windows), batch_size):
                batch = context_windows[i:i + batch_size]
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

                if i % 1000 == 0 and i > 0:
                    print(f"[dim]Processed {i}/{len(context_windows)} windows.[/dim]")
    return mlm_cache


def analyze_mlm_predictions(
        context_windows,
        mlm_cache,
        lemmatizer,
        show_examples=True
    ):

    human_probs, human_ranks, machine_probs, machine_ranks = [], [], [], []

    if show_examples:
        print("\n[bold cyan]--- MODEL PREDICTION EXAMPLES ---[/bold cyan]")

    for idx, window in enumerate(context_windows):
        window_key = f"{window['id']}_[{window['target']}]_{'_'.join(window['words'])}"

        # Retrieves the cached MLM features for the current window. 50000 is chosen arbitrarily.
        prob, rank, top_guess = mlm_cache.get(window_key, (0.0, 50000.0, ""))

        lemma = lemmatizer.lemmatize(window['target'].lower().strip(string.punctuation))
        clean_context = " ".join([w for w in window['words'] if w != '<PAD>'])

        if window.get("target_label", 0) == 0:
            human_probs.append(prob)
            human_ranks.append(rank)
        else:
            machine_probs.append(prob)
            machine_ranks.append(rank)

        if show_examples and idx < 5:
            print(f"Context: {clean_context}")
            print(f"  Target: '[bold]{window['target']}[/bold]' | Lemma: '{lemma}' | Rank: {rank} | Guess: '{top_guess}'")
            print("[dim]" + "-" * 40 + "[/dim]")

    def print_stats(name, data):
        if len(data) == 0:
            print(f"  [dim]{name:12} -> No data[/dim]")
            return
        data_arr = np.array(data)
        print(f"  [dim]{name:12}[/dim] -> [bold green]Mean:[/bold green] {data_arr.mean():.6e} | [bold blue]Median:[/bold blue] {np.median(data_arr):.6e}")

    print("\n[magenta]" + "=" * 50 + "[/magenta]")
    print("[magenta] MLM PREDICTION FEATURES ANALYSIS [/magenta]")
    print("[magenta]" + "=" * 50 + "[/magenta]")
    print(f"[bold green][ HUMAN ][/bold green] Samples: {len(human_probs)}")
    print_stats("Probability", human_probs)
    print_stats("Rank", human_ranks)
    print(f"\n[bold red][ MACHINE ][/bold red] Samples: {len(machine_probs)}")
    print_stats("Probability", machine_probs)
    print_stats("Rank", machine_ranks)
    print("[bold magenta]" + "=" * 50 + "[/bold magenta]")
