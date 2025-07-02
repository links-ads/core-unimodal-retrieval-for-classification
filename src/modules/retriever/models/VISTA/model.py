import torch
from .visual_bge import Visualized_BGE
from typing import List, Dict
from PIL import Image

class VisualizedBGEEmbeddings():
    def __init__(
        self, 
        model_name_bge: str, 
        model_weight, 
        # device: str = None
    ):

        """Initialize the model"""
        self.model = Visualized_BGE(
             model_name_bge = model_name_bge,
             model_weight= model_weight
        )

        self.model.eval()
        # # Set device
        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = self.model.to(self.device)

    def encode_corpus(
        self, 
        corpus: Dict[str, Dict[str, str]],
        c_type: str,
        **kwargs
    ) -> List[torch.Tensor]:
        """Generate embeddings for a list of texts and optional images"""
        embeddings = []
        with torch.no_grad():
            for _, corpus in enumerate(corpus):
                text, image = None, None
                if c_type in ['text', 'text-image']:
                    text = corpus['text']
                if c_type in ['image', 'text-image']:
                    image = corpus['image']
                embeddings.append(self.model.encode(text=text, image=image).tolist()[0])
        return embeddings

    def encode_queries(
        self, 
        queries: Dict[str, Dict[str, str]],
        q_type: str,
        **kwargs
    ) -> List[torch.Tensor]:
        """Generate embedding for a query text and optional image"""
        embeddings = []
        with torch.no_grad():
            for _, queries in enumerate(queries):
                text, image = None, None
                if q_type in ['text', 'text-image']:
                    text = queries['text']
                if q_type in ['image', 'text-image']:
                    image = queries['image']
                embeddings.append(self.model.encode(text=text, image=image).tolist()[0])
        return embeddings