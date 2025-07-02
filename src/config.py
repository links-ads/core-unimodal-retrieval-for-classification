import yaml
from pydantic import BaseModel
from typing import Optional, Dict, List

class DatasetColumnConfig(BaseModel):
    id: str
    rels: str
    image: str
    q_text: str
    c_text: str
    label: str

class SyntheticDatasetConfig(BaseModel):
    path: str
    column_selection: Dict[str, Dict[str, DatasetColumnConfig]]

class SMSSpamDatasetConfig(BaseModel):
    path: str
    column_selection: Dict[str, Dict[str, DatasetColumnConfig]]

class YahooAnswersTopicDatasetConfig(BaseModel):
    path: str
    column_selection: Dict[str, Dict[str, DatasetColumnConfig]]

class DatasetConfig(BaseModel):
    yahoo_answers_topic: YahooAnswersTopicDatasetConfig
    synthetic: SyntheticDatasetConfig
    sms: SMSSpamDatasetConfig
    language: str
    modality: str
    q_type: str
    c_type: str

class UndersamplingConfig(BaseModel):
    type: str
    ratio: float
    near_miss_version: int
    n_neighbors: int

class RetrieverConfig(BaseModel):
    modality: str
    k: List[int]
    use_weighted_scoring: bool
    combined_weigth1: float
    combined_weigth2: float
    name: Dict[str, str]
    batch_size: int
    score_function: str
    undersampling: UndersamplingConfig

class DecoderConfig(BaseModel):
    model_path: str
    model_type: str
    example_template: str
    prompt_template: str

class ClassifierConfig(BaseModel):
    name: Dict[str, str]
    pooling_method: str
    batch_size: int
    keywords: Dict[str, List[str]]
    
class ModelConfig(BaseModel):
    retriever: RetrieverConfig
    decoder: DecoderConfig
    classifier: ClassifierConfig

class EvaluationConfig(BaseModel):
    cv: int
    store_predictions: bool
    use_retriever: bool
    use_undersampling: bool
    use_decoder: bool

class Config(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
    evaluation: EvaluationConfig
