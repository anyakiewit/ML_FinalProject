import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


class MLMFeatureExtractor:
    def __init__(self, model_name="distilroberta-base"):

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

    def get_prediction_features(self, context_words, target_word):
        middle_idx = len(context_words) // 2

        # Replace target word with a mask
        masked_words = context_words.copy()
        masked_words[middle_idx] = self.tokenizer.mask_token

        # Remove <PAD> tokens, Model handles padding on its own.
        clean_words = []
        for word in masked_words:
            if word != "<PAD>":
                clean_words.append(word)

        masked_text = " ".join(clean_words)

        # Precaution if a context-window happens to be empty.
        inputs = self.tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=128)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Find the index of the mask token.
        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

        # Fallback
        if len(mask_token_index) == 0:
            return [0.0, 10000.0]
        
        mask_token_index = mask_token_index[0]

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract the logits, raw scores, for the mask token.
        mask_token_logits = outputs.logits[0, mask_token_index, :]

        # Convert logits to probabilities.
        mask_token_probs = torch.softmax(mask_token_logits, dim=0)

        # Tokenizes actual target word. 
        target_token_ids = self.tokenizer(target_word, add_special_tokens=False)["input_ids"]

        # If target word got split up into multiple sub-words, we take the first part.
        if not target_token_ids:
            return [0.0, 10000.0]
        first_target_token_id = target_token_ids[0]

        # Probability of actual target word.
        target_prob = mask_token_probs[first_target_token_id].item()

        # What rank was the actual target word.
        sorted_indices = torch.argsort(mask_token_probs, descending=True)

        rank = (sorted_indices == first_target_token_id).nonzero(as_tuple=True)[0].item() + 1

        return [target_prob, float(rank)]