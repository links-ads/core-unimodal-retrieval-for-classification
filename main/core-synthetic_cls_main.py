import pandas as pd
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import StratifiedShuffleSplit
import nltk
from src.logging import LoggingHandler
from src.parser import Parser
from src.datasets import CoReSyntheticDataset
from src.modules.classifier.models import BaselineClassifier, Label2TransformerVecClassifier, ZeroShotNLIClassifier
from src.evaluations.classification_evaluation import EvaluateClassification

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

parser = Parser()
config, device = parser.parse_args()

splits = None
if config.dataset.modality == "half-half":
    df = pd.read_csv(config.dataset.synthetic.path)
    ids = df[config.dataset.synthetic.column_selection[config.dataset.modality][config.dataset.language].id].tolist()
    labels = df[config.dataset.synthetic.column_selection[config.dataset.modality][config.dataset.language].label].tolist()
    shufle_splitter = StratifiedShuffleSplit(n_splits=config.evaluation.cv, test_size=0.5, random_state=0)
    splits = list(shufle_splitter.split(ids, labels))
    
for n_splits in range(config.evaluation.cv):
    dataset = CoReSyntheticDataset()
    dataset.load(
        info=config.dataset.synthetic.column_selection[config.dataset.modality][config.dataset.language],
        path_text=config.dataset.synthetic.path,
        ids_db=list(splits[n_splits][0]) if splits else None,
        ids_query=list(splits[n_splits][1]) if splits else None
    )
    dataset.preprocess(
                        lst_stopwords=nltk.corpus.stopwords.words("english"), 
                        stemm=False, 
                        lemm=True
                    )

    model = BaselineClassifier(
            model_path=config.model.classifier.name[config.dataset.language],
            pooling_method=config.model.classifier.pooling_method,
            batch_size=config.model.classifier.batch_size,

        )
    
    # model = Label2TransformerVecClassifier(
    #         encoder_path=config.model.classifier.name[config.dataset.language],
    #     )
    
    # model = ZeroShotNLIClassifier(
    #         model_name=config.model.classifier.name[config.dataset.language],
    #         batch_size=config.model.classifier.batch_size,
    #     )

    # Define the keywords
    keywords = config.model.classifier.keywords
    
    # print(keywords)
    _, predicted = model.classify(
        corpus=dataset.corpus,
        queries=dataset.queries,
        keywords=keywords,
    )
    
    # print(predicted)
    eval_dict = EvaluateClassification().evaluate_classifier(
        pred=predicted,
        true=[q['labels'] for _, q in dataset.queries.items()]
    )
