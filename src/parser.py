from argparse import ArgumentParser
from pathlib import Path
import os
import yaml

import torch

from src.config import Config


class Parser:
    def __init__(self):
        self.parser = ArgumentParser()
        # General mode args
        self.parser.add_argument(
            "-c", "--config", type=Path, required=True, dest="CONFIG")
        self.parser.add_argument("--cpu", action="store_true", dest="CPU")
        self.parser.add_argument(
            "--q_type", type=str, dest="q_type")
        self.parser.add_argument(
            "--c_type", type=str, dest="c_type")
        self.parser.add_argument(
            "--language", type=str, dest="language")
        self.parser.add_argument(
            "--modality", type=str, dest="modality")
        self.parser.add_argument(
            "--cv", type=int, dest="cv")
        self.parser.add_argument(
            "--undersampling_type", type=str, dest="undersampling_type")
        self.parser.add_argument(
            "--undersampling_ratio", type=float, dest="undersampling_ratio")
        self.parser.add_argument(
            "--undersampling_near_miss_version", type=int, dest="undersampling_near_miss_version")
        self.parser.add_argument(
            "--undersampling_n_neighbors", type=int, dest="undersampling_n_neighbors")
        self.parser.add_argument(
            "--combined_weigth1", type=float, dest="combined_weigth1")
        self.parser.add_argument(
            "--combined_weigth2", type=float, dest="combined_weigth2")
        # General data args

    def parse_args(self):
        self.args = self.parser.parse_args()

        if not self.args.CPU:
            if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
                exit("Specify CUDA_VISIBLE_DEVICES environment variable")
        available_devs = torch.cuda.device_count()
        if available_devs >= 1 and not self.args.CPU:
            device = "cuda"
        else:
            device = "cpu"

        with open(self.args.CONFIG) as f:
            d = yaml.safe_load(f)
            config = Config(**d)
              
        if self.args.q_type is not None:
            config.dataset.q_type = self.args.q_type
        
        if self.args.c_type is not None:
            config.dataset.c_type = self.args.c_type
        
        if self.args.combined_weigth1 is not None:
            config.model.retriever.combined_weigth1 = self.args.combined_weigth1
        
        if self.args.combined_weigth2 is not None:
            config.model.retriever.combined_weigth2 = self.args.combined_weigth2
        
        if self.args.language is not None:
            config.dataset.language = self.args.language
        
        if self.args.modality is not None:
            config.dataset.modality = self.args.modality
        
        if self.args.cv is not None:
            config.evaluation.cv = self.args.cv
        
        if self.args.undersampling_type is not None:
            config.model.retriever.undersampling.type = self.args.undersampling_type
        
        if self.args.undersampling_ratio is not None:
            config.model.retriever.undersampling.ratio = self.args.undersampling_ratio
        
        if self.args.undersampling_near_miss_version is not None:
            config.model.retriever.undersampling.near_miss_version = self.args.undersampling_near_miss_version
        
        if self.args.undersampling_n_neighbors is not None:
            config.model.retriever.undersampling.n_neighbors = self.args.undersampling_n_neighbors

        return config, device
