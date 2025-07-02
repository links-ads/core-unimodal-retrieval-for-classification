from __future__ import annotations

import logging
import numpy as np
from tqdm import trange
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from ..utils import cosine_sim

logger = logging.getLogger(__name__)

class BaselineClassifier:
    def __init__(
        self,
        model_path: str | tuple = None,
        sep: str = " ",
        pooling_method: str = "mean",
        batch_size: int = 8,
        **kwargs,
    ):
        self.sep = sep
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_path
            )
        self.model.eval()
        
        # Set pooling method (mean, cls, max, mean_without_special)
        supported_pooling = ["mean", "cls", "max", "mean_without_special"]
        if pooling_method not in supported_pooling:
            logger.warning(f"Unsupported pooling method: {pooling_method}. Using 'mean' instead.")
            self.pooling_method = "mean"
        else:
            self.pooling_method = pooling_method
        self.batch_size = batch_size
        
    def _pool_embeddings(self, token_embeddings, attention_mask, input_ids=None):
        """Apply the selected pooling method to the token embeddings.
        
        Args:
            token_embeddings: Model output embeddings
            attention_mask: Attention mask from tokenizer
            input_ids: Token IDs from tokenizer (needed for mean_without_special)
        """
        if self.pooling_method == "cls":
            # Use the CLS token (first token) embedding
            return token_embeddings[:, 0]
            
        elif self.pooling_method == "max":
            # Max pooling - take maximum value over token dimension
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            return torch.max(token_embeddings, 1)[0]
            
        elif self.pooling_method == "mean_without_special":
            # Mean pooling without CLS and SEP tokens
            # Create mask for all tokens except CLS (first) and SEP (identified by tokenizer.sep_token_id)
            batch_size, seq_len, emb_dim = token_embeddings.size()
            
            # Create a mask that excludes special tokens
            # Start with attention mask
            mask = attention_mask.clone().detach()
            
            # Set CLS token (index 0) to 0 for all sequences
            mask[:, 0] = 0
            
            # Find SEP tokens and set them to 0
            if input_ids is not None:
                for i in range(batch_size):
                    # Find positions where SEP token occurs
                    sep_positions = (input_ids[i] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                    for pos in sep_positions:
                        mask[i, pos] = 0
                    
            # Expand mask for embedding dimension
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Sum embeddings with mask applied (excluding special tokens)
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            
            # Calculate sum of mask for proper averaging (avoid division by zero)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            return sum_embeddings / sum_mask
            
        else:  # Default: mean pooling
            # Mean pooling - take attention mask into account for averaging
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
    
    def _encode_text(
        self, 
        sentences: list[str],
        batch_size: int = 8, 
        **kwargs
    ) -> tuple[np.ndarray, dict]:

        # Create a single embedding matrix for all sentences
        all_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(sentences), batch_size):
                sub_sentences = [
                sentence for sentence in sentences[start_idx : start_idx + batch_size]
                ]
                ctx_input = self.tokenizer(sub_sentences, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
                
                # Get the model output and convert to embeddings
                model_output = self.model(**ctx_input)
                
                # Get the last_hidden_state from the output
                token_embeddings = model_output.last_hidden_state
                
                # Apply selected pooling method
                batch_embeddings = self._pool_embeddings(
                    token_embeddings, 
                    ctx_input['attention_mask'],
                    ctx_input['input_ids'] if self.pooling_method == "mean_without_special" else None
                )
                
                # Convert to numpy and append to list
                all_embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batch embeddings
        embs = np.vstack(all_embeddings)
        return embs

    def _encode_labels(
        self,
        keywords: dict[str, list[str]],
        batch_size: int = 8,
        **kwargs,
    ):      
        # Create embeddings for keywords
        labels_embs = {}
        for k, v in keywords.items():
            # Process each word in the keyword list
            word_embeddings = []
            for word in v:
                # Tokenize and encode
                word_input = self.tokenizer(word, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    word_output = self.model(**word_input)
                    word_vectors = word_output.last_hidden_state
                    
                    # Apply selected pooling method
                    word_embedding = self._pool_embeddings(
                        word_vectors, 
                        word_input['attention_mask'],
                        word_input['input_ids'] if self.pooling_method == "mean_without_special" else None
                    )
                    
                    word_embeddings.append(word_embedding.cpu().numpy())
            
            # Average all word embeddings for this keyword/category
            labels_embs[k] = np.mean(np.vstack(word_embeddings), axis=0)
        return labels_embs

    def classify(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        keywords: dict[str, str],
    ): 
        sentences = [query['text'] for _, query in queries.items()]
        # get embeddings
        embs= self._encode_text(
            sentences,
            batch_size=self.batch_size,
        )
        
        labels_embs = self._encode_labels(
            keywords,
            batch_size=self.batch_size,
        )
        
        # get the labels
        predicted_prob = np.array([cosine_sim(embs, y).T.tolist()[0] for y in labels_embs.values()]).T
        labels = list(labels_embs.keys())
        
        ## adjust and rescale
        for i in range(len(predicted_prob)):
            ### assign randomly if there is no similarity
            if sum(predicted_prob[i]) == 0:
                predicted_prob[i] = [0]*len(labels)
                predicted_prob[i][np.random.choice(range(len(labels)))] = 1
            ### rescale so they sum=1
            predicted_prob[i] = predicted_prob[i] / sum(predicted_prob[i])
        
        predicted = [labels[np.argmax(pred)] for pred in predicted_prob]
        return predicted_prob, predicted