
from transformers import pipeline
from tqdm import tqdm
from datasets import Dataset
import torch

class ZeroShotNLIClassifier:
    def __init__(self, 
                 model_name: str,
                 batch_size: int = 8,
    ):
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        self.batch_size = batch_size

    # def classify(self, corpus, queries, keywords):
    #     # q_sentences = [query['text'] for _, query in queries.items()]
    #     keywords_list = [v for k, v in keywords.items()]
    #     output = self.classifier(queries, keywords_list, multi_label=False)
    #     predicted = [out["labels"][0] for out in output]
    #     pred_probs = [out["scores"][0] for out in output]
    #     return pred_probs, predicted
    
    # def classify(self, corpus, queries, keywords):
    #     # q_sentences = [query['text'] for _, query in queries.items()]
    #     keywords_list = [v for k, v in keywords.items()]
        
    #     # Create list to store results
    #     output = []
        
    #     # Add progress bar
    #     for query in tqdm(queries, desc="NLI Classification", unit="query"):
    #         result = self.classifier(query, keywords_list, multi_label=False)
    #         output.append(result)
            
    #     predicted = [out["labels"][0] for out in output]
    #     # replace label predicted with the key of the keywords
    #     predicted = [list(keywords.keys())[list(keywords.values()).index(label)] for label in predicted]
    #     pred_probs = [out["scores"][0] for out in output]
    #     return pred_probs, predicted

    def classify(self, corpus, queries, keywords):
        keywords_list = [v for k, v in keywords.items()]
        
        # Convert queries to a format suitable for batching
        # queries_text = [query for query in queries]
        
        # Create a HuggingFace Dataset
        dataset = Dataset.from_dict({"text": queries})
        
        # Process in batches
        outputs = []
        for batch in tqdm(dataset.iter(batch_size=self.batch_size), 
                          desc="NLI Classification", 
                          total=len(dataset)//self.batch_size + (1 if len(dataset) % self.batch_size > 0 else 0)):
            results = self.classifier(batch["text"], keywords_list, multi_label=False)
            outputs.extend(results)
        
        predicted = [out["labels"][0] for out in outputs]
        # Replace label predicted with the key of the keywords
        predicted = [list(keywords.keys())[list(keywords.values()).index(label)] for label in predicted]
        pred_probs = [out["scores"][0] for out in outputs]
        
        return pred_probs, predicted
