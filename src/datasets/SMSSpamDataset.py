import logging
import pandas as pd

from typing import List
from .utils import preprocess_text

logger = logging.getLogger(__name__)


class SMSSpamDataset:
    def __init__(self):   
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

    def load(
        self,
        info: dict[str, str],
        path_text: str, 
        path_image: str = None, 
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
            # corpus is a dict {'uuid': {'text': 'text', 'labels': label, 'image': 'image'}}
            id = str(row[info.id])
            self.corpus[id] = {
                'text': row[info.c_text], 
                'labels': row[info.label]
            }
        # create queries
        for _, row in df_query.iterrows():
            # queries is a dict {'uuid': {'text': 'text', 'image': 'image'}}
            # unique uuid for each query from library
            id = str(row[info.id])
            self.queries[id] =  {
                                    'text': row[info.q_text], 
                                    'labels': row[info.label]
                                }

        for _, row in df_query.iterrows():
            # qrels is a dict {'uuid_query': {'uuid_corpus': 1}}
            id = str(row[info.id])
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
