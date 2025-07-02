import pandas as pd
import logging
import os
import sys
from sklearn.datasets import fetch_20newsgroups

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logging import LoggingHandler
from src.parser import Parser
from src.datasets import NewsGroupsDataset
from sklearn.model_selection import StratifiedShuffleSplit
from src.modules.retriever.models import SentenceBERT
from src.evaluations.classification_evaluation import EvaluateClassification
from src.modules.retriever.search.dense.search import DenseRetrievalExactSearch as DRES
from src.modules.decoder.models.LlamaCpp import LlamaCpp
from src.modules.decoder.generate import ClassifyQueryWithGenerator

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

parser = Parser()
config, device = parser.parse_args()
scores = {'retriever': {}, 'reranker': {}, 'decoder': {}}

# Create a list of all document IDs
all_docs = [f"doc_{i}" for i in range(18846)]  # Total size of 20newsgroups dataset
all_labels = fetch_20newsgroups(subset='train').target_names
categories_to_use = None  # Use all categories, or specify a subset if needed

# Create splits using stratified sampling
labels = fetch_20newsgroups(subset='train').target.tolist() + fetch_20newsgroups(subset='test').target.tolist()
shuffle_splitter = StratifiedShuffleSplit(n_splits=config.evaluation.cv, test_size=0.5, random_state=0)
splits = list(shuffle_splitter.split(all_docs, labels))

retriever_model = SentenceBERT(
    model_path=config.model.retriever.name[config.dataset.language]
)
print(f"Using retriever model: {config.model.retriever.name[config.dataset.language]}")
retriever = DRES(
                model=retriever_model,
                batch_size=config.model.retriever.batch_size,
                corpus_chunk_size=512 * 9999,
            )
        
if config.evaluation.use_decoder:
    decoder = LlamaCpp(
                        model_path=config.model.decoder.model_path,
                        model_type=config.model.decoder.model_type,
                    )
    classifier = ClassifyQueryWithGenerator(
                                            generator=decoder,
                                            example_template=config.model.decoder.example_template,
                                            prompt_template=config.model.decoder.prompt_template
                                        )
        
for n_splits in range(config.evaluation.cv):
    # get db and query
    dataset = NewsGroupsDataset()
    dataset.load(
        categories=categories_to_use,
        ids_db=[all_docs[i] for i in splits[n_splits][0]],
        ids_query=[all_docs[i] for i in splits[n_splits][1]]
    )
    # dataset.downsample_corpus(
    #     retriever=retriever
    # )
    
    results = retriever.search(
        corpus=dataset.corpus, 
        queries=dataset.queries, 
        top_k=max(config.model.retriever.k), 
        score_function=config.model.retriever.score_function
    )
    
    eval_dict = EvaluateClassification().evaluate_retriever(
                                                            dataset=dataset, 
                                                            results=results, 
                                                            k=config.model.retriever.k,
                                                            use_weighted_scoring=config.model.retriever.use_weighted_scoring
                                                            )
    for k in eval_dict:
        if k not in scores['retriever']:
            scores['retriever'][k] = []
        scores['retriever'][k].append(eval_dict[k])    

    if config.evaluation.use_decoder:
        # results is a dict of dicts {'uuid_query': {k, label}}
        decoder_results = classifier.classify(
                                        similar_corpus=None,
                                        query=dataset.queries,
                                        corpus=dataset.corpus,
                                        k=config.model.retriever.k,
                                    )
        decoder_results = {k: v for k, v in decoder_results.items() if k in decoder_results}
        
        eval_dict = EvaluateClassification().evaluate_decoder(
            dataset=dataset,
            results=decoder_results,
            k=config.model.retriever.k
        )
        for k in eval_dict:
            if k not in scores['decoder']:
                scores['decoder'][k] = []
            scores['decoder'][k].append(eval_dict[k])

### Evaluation       
logging.info(f"CV Results")
logging.info(f"=====================")
logging.info("Retrieval results: ")
for k in scores['retriever']:
    # average f1
    avg_f1 = sum([score['macro_f1'] for score in scores['retriever'][k]]) / len(scores['retriever'][k])
    logging.info(f"Average F1 for retriever@{k}: {avg_f1}")

if config.evaluation.use_decoder:
    logging.info(f"=====================")
    logging.info("Decoder results: ")
    for k in scores['decoder']:
        # average f1
        avg_f1 = sum([score['macro_f1'] for score in scores['decoder'][k]]) / len(scores['decoder'][k])
        logging.info(f"Average F1 for decoder@{k}: {avg_f1}")