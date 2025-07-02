from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class CombinedRetriever:
    """
    A class that combines two retrievers (VISTA and exact) using min-max normalization
    """
    def __init__(
        self,
        retriever1,  # VISTA retriever
        retriever2,  # Exact retriever
        weight1: float = 0.6,  # VISTA weight
        weight2: float = 0.4,  # Exact weight
    ):
        """
        Initialize combined retriever with two pre-initialized retrievers

        Args:
            retriever1: First retriever
            retriever2: Second retriever
            weight1: Weight for first retriever
            weight2: Weight for second retriever
        """
        self.retriever1 = retriever1
        self.retriever2 = retriever2
        self.weight1 = weight1
        self.weight2 = weight2
        
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Perform min-max normalization on scores"""
        scores = [float(s) for s in scores]
        if len(scores) == 0 or all(s == scores[0] for s in scores):
            return [0.0] * len(scores)
        min_score = min(scores)
        max_score = max(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform search using both retrievers and combine results
        """
        # Get results from first retriever (VISTA)
        results1 = self.retriever1.search(
            corpus=corpus,
            queries=queries,
            top_k=top_k,
            **kwargs
        )

        # Get results from second retriever (Exact)
        results2 = self.retriever2.search(
            corpus=corpus,
            queries=queries,
            top_k=top_k,
            q_type="text",
            c_type="text"
        )

        # Combine results
        combined_results = {}
        for query_id in queries.keys():
            # Convert scores to float
            dict1 = {k: float(v) for k, v in results1[query_id].items()}
            dict2 = {k: float(v) for k, v in results2[query_id].items()}
            
            # Get all unique document IDs
            all_docs = set(dict1.keys()) | set(dict2.keys())
            
            # Normalize scores for each retriever
            scores1 = self._normalize_scores(list(dict1.values()))
            scores2 = self._normalize_scores(list(dict2.values()))
            
            # Create normalized score dictionaries
            norm1 = dict(zip(dict1.keys(), scores1))
            norm2 = dict(zip(dict2.keys(), scores2))
            
            # Combine scores
            combined_dict = {}
            for doc_id in all_docs:
                score1 = norm1.get(doc_id, 0.0)
                score2 = norm2.get(doc_id, 0.0)
                combined_dict[doc_id] = float(self.weight1 * score1 + self.weight2 * score2)
            
            combined_results[query_id] = combined_dict

        return combined_results
