import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


class MLMFeatureExtractor:
    def __init__(self, model_name="roberta-large"):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()

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

    def get_prediction_features_batch(self, batch_context_words, batch_target_words):
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

        for i in range(len(batch_masked_texts)):
            mask_token_index = torch.where(inputs["input_ids"][i] == self.tokenizer.mask_token_id)[0]

            if len(mask_token_index) == 0:
                results.append([0.0, 50000.0, ""])
                continue

            mask_token_index = mask_token_index[0].item()
            mask_token_logits = outputs.logits[i, mask_token_index, :].clone()

            mask_token_logits[self.tokenizer.all_special_ids] = -float('inf')
            mask_token_probs = torch.softmax(mask_token_logits, dim=0)

            clean_target = batch_target_words[i].strip(string.punctuation)
            target_token_ids = self.tokenizer(" " + clean_target, add_special_tokens=False)["input_ids"]

            if not target_token_ids:
                results.append([0.0, 50000.0, ""])
                continue

            total_log_prob = 0.0
            best_rank = 50000.0

            for token_id in target_token_ids:
                prob = mask_token_probs[token_id].item()
                
                total_log_prob += math.log(max(prob, 1e-10))

                rank = (mask_token_probs > prob).sum().item() + 1

                if rank < best_rank:
                    best_rank = rank

            avg_log_prob = total_log_prob / len(target_token_ids)
            target_prob = math.exp(avg_log_prob)

            top_token_id = torch.argmax(mask_token_probs).item()
            top_guess = self.tokenizer.decode([top_token_id]).strip()

            results.append([target_prob, float(best_rank), top_guess])

        return results