import pandas as pd
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logging import LoggingHandler
from src.parser import Parser
from src.datasets import SMSSpamDataset
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

splits = None
if config.dataset.modality == "half-half":
    df = pd.read_csv(config.dataset.sms.path)
    ids = df[config.dataset.sms.column_selection[config.dataset.modality][config.dataset.language].id].tolist()
    labels = df[config.dataset.sms.column_selection[config.dataset.modality][config.dataset.language].label].tolist()
    shufle_splitter = StratifiedShuffleSplit(n_splits=config.evaluation.cv, test_size=0.5, random_state=0)
    splits = list(shufle_splitter.split(ids, labels))

retriever_model = SentenceBERT(
    model_path=config.model.retriever.name[config.dataset.language]
)

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
    dataset = SMSSpamDataset()
    dataset.load(
        info=config.dataset.sms.column_selection[config.dataset.modality][config.dataset.language],
        path_text=config.dataset.sms.path,
        ids_db=list(splits[n_splits][0]) if splits else None,
        ids_query=list(splits[n_splits][1]) if splits else None
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
        decoder_results = {k: "spam" if v else "ham" for k, v in decoder_results.items() if k in decoder_results}
        
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
