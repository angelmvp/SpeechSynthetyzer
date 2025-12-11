from prosodiamodules.modelBertProsodia import Bert
from pytorch_transformers import BertTokenizer
import torch
import os
import json
class Prosodia_module():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tag_to_index, self.index_to_tag = self.read_tags()
        model_path = "./prosodiamodules/elbueno.pt"
        self.model = Bert(self.device, labels=len(self.tag_to_index))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    def read_tags(self):
        if not os.path.exists('./prosodiamodules/index_to_tag.json'):
            print("no existe index_to_tag.json")
            return None
        with open('./prosodiamodules/index_to_tag.json', 'r') as f:
            # JSON loads string keys, convert them back to int
            loaded_dict_str_keys = json.load(f)
            index_to_tag= {int(k): v for k, v in loaded_dict_str_keys.items()}
        if not os.path.exists('./prosodiamodules/tag_to_index.json'):
            print("no existe tag_to_index.json")
            return None
        with open('./prosodiamodules/tag_to_index.json', 'r') as f:
            tag_to_index= json.load(f)
        return tag_to_index, index_to_tag


    def obtener_indices_prosodia(self, words_array):
        """
        Takes an array of English words and returns an array of corresponding prosodic tags.

        Args:
            words_array (list): A list of English words (e.g., ["hello", "world"]).

        Returns:
            list: A list of strings, where each string is the prosodic tag for the
                corresponding input word.
        """
        # Ensure the predictor components are set up
        model_infer = self.model
        device_infer = self.device
        tokenizer = self.tokenizer
        index_to_tag_infer = self.index_to_tag
        # 1. Tokenize and prepare input
        all_input_ids = []
        original_word_main_piece_indices = [] # Stores index in tokenized sequence where an original word's main piece starts

        # Add [CLS] token
        all_input_ids.append(tokenizer.convert_tokens_to_ids(["[CLS]"])[0])
        current_token_idx = 1 # Start after [CLS]

        for word in words_array:
            tokens = tokenizer.tokenize(word)
            print(word,tokens)
            if not tokens: # Handle words that might not tokenize (e.g., empty string)
                continue

            word_ids = tokenizer.convert_tokens_to_ids(tokens)
            all_input_ids.extend(word_ids)

            # Keep track of the starting index of the first subword token for each original word (main piece index).
            original_word_main_piece_indices.append(current_token_idx)
            current_token_idx += len(tokens)

        # Add [SEP] token
        all_input_ids.append(tokenizer.convert_tokens_to_ids(["[SEP]"])[0])
        print(all_input_ids)
        # Convert the list of numerical IDs into a PyTorch tensor, and create an attention mask tensor
        input_ids_tensor = torch.tensor([all_input_ids]).to(device_infer)
        attention_mask_tensor = torch.tensor([[1] * len(all_input_ids)]).to(device_infer) # All tokens are attended to

        # 2. Perform Inference (dummy y is needed for model's forward method signature)
        with torch.no_grad():
            dummy_y = torch.zeros_like(input_ids_tensor).to(device_infer) # Create a dummy 'y' tensor
            logits, _, y_hat_tokens = model_infer(input_ids_tensor, dummy_y) # y_hat_tokens contains token-level predictions

        # 3. Extract Prosodic Indices for original words
        predicted_indices_all_tokens = y_hat_tokens.squeeze(0).cpu().numpy() # Convert to numpy array

        prosodic_tags = []
        # Loop through the original words' main piece indices
        i=-1
        for main_piece_idx in original_word_main_piece_indices:
            i+=1
            if 0 <= main_piece_idx < len(predicted_indices_all_tokens):
                predicted_tag_idx = predicted_indices_all_tokens[main_piece_idx]
                tag = index_to_tag_infer.get(predicted_tag_idx, "NA") # Use "NA" for unknown tags
                print(f"Word index {main_piece_idx}  word { words_array[i]} predicted tag index {predicted_tag_idx} tag {tag}")
                prosodic_tags.append(tag)
            else:
                prosodic_tags.append("NA") # Should not happen if `main_piece_idx` is within bounds

        return prosodic_tags

print("The `obtener_indices_prosodia` function is prepared and the tokenizer will be initialized upon its first call.")

prosodia = Prosodia_module()
print("PROSODIA MODULE INITIALIZED")
tokens = ['hello','world',',','where','do','you','from','?', 'I', 'am', 'from', 'bangladesh','.' 'and ','you','?']
prosodic_tags = prosodia.obtener_indices_prosodia(tokens)
print(prosodic_tags)