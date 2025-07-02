import logging
import pandas as pd
import nltk
import torch

from typing import List
from .utils import preprocess_text
from src.modules.retriever.search.dense.search import DenseRetrievalExactSearch as DRES
from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours

logger = logging.getLogger(__name__)


class CoReSyntheticDataset:
    def __init__(self):   
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

    def load(
        self,
        info: dict[str, str],
        path_text: str, 
        ids_db: List[str] = None, 
        ids_query: List[str] = None
    ):
        
        """
        Load the dataset and return the db and query dataframes
        
        Args:
        - info: dict containing the name of the columns for the text and image of the query and db
        - path_text: path to the text file
        - path_image: path to the image folder
        - ids_db: list of ids of the db
        - ids_query: list of ids of the query
        """
        
        df = pd.read_csv(path_text)
        # split query and corpus or not
        if ids_db and ids_query:
            df_db = df[df.index.isin(ids_db)]
            df_query = df[df.index.isin(ids_query)]
            df_db.reset_index(drop=True, inplace=True)
            df_query.reset_index(drop=True, inplace=True)
        else:
            df_db = df
            df_query = df
        
        # create corpus
        for _, row in df_db.iterrows():
            # corpus is a dict {'uuid': {'text': 'text', 'labels': label}}
            id = row[info.id]
            self.corpus[id] = {
                'text': row[info.c_text], 
                'labels': row[info.label]
            }
        # create queries
        for _, row in df_query.iterrows():
            # queries is a dict {'uuid': {'text': 'text', 'image': 'image'}}
            # unique uuid for each query from library
            id = row[info.id]
            self.queries[id] =  {
                                    'text': row[info.q_text], 
                                    'labels': row[info.label]
                                }

        for _, row in df_query.iterrows():
            # qrels is a dict {'uuid_query': {'uuid_corpus': 1}}
            id = row[info.id]
            self.qrels[id] = {row[info.rels]: 1}
    
    def preprocess(
        self,
        lst_regex: list = None, 
        punkt: bool = True, 
        lower: bool = True, 
        slang: bool = True, 
        lst_stopwords: bool = None, 
        stemm: bool = False, 
        lemm: bool = False
    ):
        # Preprocess the corpus and queries
        for k, v in self.corpus.items():
            v['labels'] = "Patchable" if v['labels'] else "Not Patchable"
            v['text'] = preprocess_text(
                                    txt = v['text'], 
                                    punkt = punkt, 
                                    lower = lower, 
                                    slang = slang, 
                                    lst_regex=lst_regex, 
                                    lst_stopwords=lst_stopwords, 
                                    stemm=stemm, 
                                    lemm=lemm
                                )
        
        for k, v in self.queries.items():
            v['labels'] = "Patchable" if v['labels'] else "Not Patchable"
            v['text'] = preprocess_text(
                                            txt = v['text'], 
                                            punkt = punkt, 
                                            lower = lower, 
                                            slang = slang, 
                                            lst_regex=lst_regex, 
                                            lst_stopwords=lst_stopwords, 
                                            stemm=stemm, 
                                            lemm=lemm
                                        )
    
    def downsample_corpus(self, retriever: DRES, elbow_value: float = 18.52):
        # get corpus with label positive
        positive_corpus = {k: v for k, v in self.corpus.items() if v['labels'] == 1}
        print(f"Positive samples in corpus: {len(positive_corpus)}")
        if not positive_corpus:
            print("No positive samples found in corpus. Skipping downsampling.")
            return
        # get neighborhood, results is a dicts of dicts {'uuid_query': {'uuid_corpus': score, ...}}
        results = retriever.search(
                                    corpus=self.corpus,
                                    queries=positive_corpus,
                                    top_k=len(self.corpus),
                                    score_function="l2",
                                )
        to_drop = set()
        for query_id, query_results in results.items():
            # Check if query_results is a dictionary
            if not isinstance(query_results, dict) or not query_results:
                print(f"Skipping empty or invalid results for query {query_id}")
                continue
                
            # Create DataFrame with explicit index
            neighborhood = pd.DataFrame.from_dict(query_results, orient='index')
            neighborhood.columns = [query_id]
            
            # get the scores
            neighborhood['scores'] = neighborhood[query_id]
            # get the ids
            neighborhood['ID'] = neighborhood.index
            
            # Create a labels column with NaN values first
            neighborhood['labels'] = pd.NA
            
            # Then update only the valid IDs with their labels
            for idx in neighborhood.index:
                if idx in self.corpus:
                    neighborhood.loc[idx, 'labels'] = self.corpus[idx]['labels']
                    
            # Filter out rows with NaN labels
            neighborhood = neighborhood.dropna(subset=['labels'])
            
            # consider only neighborhood with label False (0)
            neighborhood = neighborhood[neighborhood['labels'] == 0]
            
            # filter the documents with score <= elbow_value
            neighborhood = neighborhood[neighborhood['scores'] >= -elbow_value]
            
            if len(neighborhood) > 0:
                to_drop.update(neighborhood['ID'].tolist())
        print(f"Total samples to remove: {len(to_drop)}")
        print(f"Samples to remove: {to_drop}")
        # remove the samples from corpus
        for k in to_drop:
            if k in self.corpus:
                del self.corpus[k]

    def random_undersample_corpus(self, retriever: DRES, random_undersampler_sampling_strategy: float = 0.02,):
        """
        Undersample the corpus using embeddings to balance the dataset.
        
        Args:
            retriever: Dense retrieval model to create embeddings
            random_undersampler_sampling_strategy: The sampling strategy to use for RandomUnderSampler
        """
        # get embedding of corpus
        print("Number of samples in corpus: ", len(self.corpus))
        print("Getting corpus embeddings...")
        
        corpus_ids = sorted(
            self.corpus,
            key=lambda k: len(self.corpus[k].get("title", "") + self.corpus[k].get("text", "")),
            reverse=True,
        )
        corpus = [self.corpus[cid] for cid in corpus_ids]
        
        corpus_embeddings = retriever.model.encode_corpus(corpus, c_type="text")
        
        # get labels
        labels = [c['labels'] for c in corpus]
        print(f"{labels.count(False)} samples with label Not Patchable")
        print(f"{labels.count(True)} samples with label Patchable")
        
        # Create a mapping from original indices to corpus_ids for later use
        index_to_id = {i: corpus_ids[i] for i in range(len(corpus_ids))}
        
        # Use RandomUnderSampler on embeddings with labels
        undersample = RandomUnderSampler(sampling_strategy=random_undersampler_sampling_strategy, random_state=0)
        
        # Fit and resample
        # We need to use a 2D array for X, so reshape if needed
        if len(corpus_embeddings.shape) == 2:
            X_resampled, y_resampled = undersample.fit_resample(corpus_embeddings, labels)
        else:
            # If embeddings aren't already 2D, reshape them
            reshaped_embeddings = corpus_embeddings.reshape(corpus_embeddings.shape[0], -1)
            X_resampled, y_resampled = undersample.fit_resample(reshaped_embeddings, labels)
        
        print(f"After undersampling: {len(X_resampled)} samples")
        
        # Get the indices from the original data that were kept in the resampled data
        # We need to use fit_sample because we need the sample_indices_ attribute
        undersample.fit_resample(corpus_embeddings, labels)
        kept_indices = undersample.sample_indices_
        
        # Create new corpus using the kept indices
        resampled_corpus = {}
        for i in kept_indices:
            id = index_to_id[i]
            resampled_corpus[id] = {
                'text': self.corpus[id]['text'],
                'labels': y_resampled[list(kept_indices).index(i)]  # Get the corresponding label
            }
        
        self.corpus = resampled_corpus
        print(f"Number of samples in corpus: {len(self.corpus)}")
        
    def near_miss_undersample_corpus(self, 
                                     retriever: DRES, 
                                     sampling_strategy: float = 0.02,
                                     near_miss_version: int = 1,
                                     n_neighbors: int = 3
                                    ):
        """
        Undersample the corpus using embeddings to balance the dataset.
        
        Args:
            retriever: Dense retrieval model to create embeddings
            sampling_strategy: The sampling strategy to use for NearMiss
            near_miss_version: The version of NearMiss to use (1, 2, or 3)
            n_neighbors: The number of neighbors to use for NearMiss
        """
        # get embedding of corpus
        print("Number of samples in corpus: ", len(self.corpus))
        print("Getting corpus embeddings...")
        
        corpus_ids = sorted(
            self.corpus,
            key=lambda k: len(self.corpus[k].get("title", "") + self.corpus[k].get("text", "")),
            reverse=True,
        )
        corpus = [self.corpus[cid] for cid in corpus_ids]
        
        corpus_embeddings = retriever.model.encode_corpus(corpus, c_type="text")
        
        # get labels
        labels = [c['labels'] for c in corpus]
        print(f"{labels.count(False)} samples with label Not Patchable")
        print(f"{labels.count(True)} samples with label Patchable")
        
        # Create a mapping from original indices to corpus_ids for later use
        index_to_id = {i: corpus_ids[i] for i in range(len(corpus_ids))}
        
        # Use RandomUnderSampler on embeddings with labels
        undersample = NearMiss(sampling_strategy=sampling_strategy, 
                               version=near_miss_version, 
                               n_neighbors=n_neighbors, 
                               n_neighbors_ver3=n_neighbors, 
                            )

        # Fit and resample
        # We need to use a 2D array for X, so reshape if needed
        if len(corpus_embeddings.shape) == 2:
            X_resampled, y_resampled = undersample.fit_resample(corpus_embeddings, labels)
        else:
            # If embeddings aren't already 2D, reshape them
            reshaped_embeddings = corpus_embeddings.reshape(corpus_embeddings.shape[0], -1)
            X_resampled, y_resampled = undersample.fit_resample(reshaped_embeddings, labels)
        
        print(f"After undersampling: {len(X_resampled)} samples")
        
        # Get the indices from the original data that were kept in the resampled data
        # We need to use fit_sample because we need the sample_indices_ attribute
        undersample.fit_resample(corpus_embeddings, labels)
        kept_indices = undersample.sample_indices_
        
        # Create new corpus using the kept indices
        resampled_corpus = {}
        for i in kept_indices:
            id = index_to_id[i]
            resampled_corpus[id] = {
                'text': self.corpus[id]['text'],
                'labels': y_resampled[list(kept_indices).index(i)]  # Get the corresponding label
            }
        
        self.corpus = resampled_corpus
        print(f"Number of samples in corpus: {len(self.corpus)}")

    def edited_nn_undersample_corpus(self, retriever: DRES, sampling_strategy: float = 0.02, n_neighbors: int = 3):
        """
        Undersample the corpus using embeddings to balance the dataset.
        
        Args:
            retriever: Dense retrieval model to create embeddings
            sampling_strategy: The sampling strategy to use for EditedNearestNeighbours
            n_neighbors: The number of neighbors to use for EditedNearestNeighbours
        """
        # get embedding of corpus
        print("Number of samples in corpus: ", len(self.corpus))
        print("Getting corpus embeddings...")
        
        corpus_ids = sorted(
            self.corpus,
            key=lambda k: len(self.corpus[k].get("title", "") + self.corpus[k].get("text", "")),
            reverse=True,
        )
        corpus = [self.corpus[cid] for cid in corpus_ids]
        
        corpus_embeddings = retriever.model.encode_corpus(corpus, c_type="text")
        
        # get labels
        labels = [c['labels'] for c in corpus]
        print(f"{labels.count(False)} samples with label Not Patchable")
        print(f"{labels.count(True)} samples with label Patchable")
        
        # Create a mapping from original indices to corpus_ids for later use
        index_to_id = {i: corpus_ids[i] for i in range(len(corpus_ids))}

        # Use EditedNearestNeighbours on embeddings with labels
        undersample = EditedNearestNeighbours(
                                                # sampling_strategy=sampling_strategy, 
                                                n_neighbors=n_neighbors
                                            )
        # Fit and resample
        # We need to use a 2D array for X, so reshape if needed
        if len(corpus_embeddings.shape) == 2:
            X_resampled, y_resampled = undersample.fit_resample(corpus_embeddings, labels)
        else:
            # If embeddings aren't already 2D, reshape them
            reshaped_embeddings = corpus_embeddings.reshape(corpus_embeddings.shape[0], -1)
            X_resampled, y_resampled = undersample.fit_resample(reshaped_embeddings, labels)
        
        print(f"After undersampling: {len(X_resampled)} samples")
        
        # Get the indices from the original data that were kept in the resampled data
        # We need to use fit_sample because we need the sample_indices_ attribute
        undersample.fit_resample(corpus_embeddings, labels)
        kept_indices = undersample.sample_indices_
        
        # Create new corpus using the kept indices
        resampled_corpus = {}
        for i in kept_indices:
            id = index_to_id[i]
            resampled_corpus[id] = {
                'text': self.corpus[id]['text'],
                'labels': y_resampled[list(kept_indices).index(i)]  # Get the corresponding label
            }
        
        self.corpus = resampled_corpus
        print(f"Number of samples in corpus: {len(self.corpus)}")
