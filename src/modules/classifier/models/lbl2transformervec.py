from lbl2vec import Lbl2TransformerVec
from transformers import AutoModel
import torch
import random

class Label2TransformerVecClassifier():
    """
    This class is used to convert labels to a transformer vector.
    """

    def __init__(self, encoder_path: str):
        self.encoder = AutoModel.from_pretrained(
                pretrained_model_name_or_path=encoder_path,
            )
        
    def classify(
        self, 
        corpus: dict[str, dict[str, str]], 
        queries: dict[str, dict[str, str]], 
        keywords: list[str]
    ):
        """
        Classify the corpus using the queries and keywords.
        
        Args:
            corpus: Dictionary containing the corpus data
            queries: Dictionary containing the query data
            keywords: List of keywords to classify
        
        Returns:
            score_similarity: List of similarity scores
            predicted: List of predicted labels
        """
        
        keywords_list = [v for k, v in keywords.items()]
        q_sentences = [query['text'] for _, query in queries.items()]    
        true_docs = [corpus[doc_id]['text'] for doc_id in corpus.keys() 
                    if corpus[doc_id]['labels'] == list(keywords.keys())[0]]
        false_docs = [corpus[doc_id]['text'] for doc_id in corpus.keys() 
                    if corpus[doc_id]['labels'] == list(keywords.keys())[1]]
        random.seed(42)
        true_sample = random.sample(true_docs, 2) if true_docs else []
        false_sample = random.sample(false_docs, 2) if false_docs else []
        
        # Convert to dictionary format for merging
        true_corpus = {f"true_{i}": {"text": doc} for i, doc in enumerate(true_sample)}
        false_corpus = {f"false_{i}": {"text": doc} for i, doc in enumerate(false_sample)}
        
        # Merge the dictionaries
        corpus_samples = {**true_corpus, **false_corpus}
        
        # Extract just the text for the c_sentences
        c_sentences = [doc["text"] for doc in corpus_samples.values()]
        
        # Initialize Label2TransformerVec
        label2vec = Lbl2TransformerVec(
            # transformer_model=self.encoder,
            keywords_list=keywords_list,
            label_names=list(keywords.keys()),
            documents=c_sentences,
            # clean_outliers=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Convert labels to transformer vectors
        label2vec.fit()
        
        # Classify the queries
        predicted = label2vec.predict_new_docs(
            documents=q_sentences, 
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # keywords_k = list(keywords.keys())
        print(label2vec.predict_model_docs().head())
        print(label2vec.predict_model_docs()['most_similar_label'].value_counts())
        print(predicted["most_similar_label"].value_counts())
        score_similarity = predicted["highest_similarity_score"].tolist()
        predicted = predicted["most_similar_label"].to_list()
        return score_similarity, predicted
