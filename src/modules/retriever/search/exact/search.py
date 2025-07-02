
import logging
from typing import Dict, List, Tuple
import heapq
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class ExactMatchRetriever:
    """
    A BEIR-style retriever that uses exact string matching for retrieval.
    Returns top-k documents based on exact match scores.
    """
    
    def __init__(self, 
                 match_type: str = "exact", 
                 case_sensitive: bool = False,
                 **kwargs):
        """
        Initialize the exact match retriever.
        
        Args:
            match_type: Type of matching ("exact", "substring", "fuzzy")
            case_sensitive: Whether to perform case-sensitive matching
        """
        self.match_type = match_type
        self.case_sensitive = case_sensitive
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.results = {}
        
        # Score functions for different match types
        self.score_functions = {
            "exact": self._exact_match_score,
            "substring": self._substring_match_score,
            "fuzzy": self._fuzzy_match_score
        }
        
        if match_type not in self.score_functions:
            raise ValueError(f"match_type must be one of: {list(self.score_functions.keys())}")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not self.case_sensitive:
            text = text.lower()
        return text.strip()
    
    def _exact_match_score(self, query: str, document: str) -> float:
        """
        Calculate exact match score.
        Returns 1.0 for exact match, 0.0 otherwise.
        """
        query_norm = self._normalize_text(query)
        doc_norm = self._normalize_text(document)
        
        if query_norm == doc_norm:
            return 1.0
        return 0.0
    
    def _substring_match_score(self, query: str, document: str) -> float:
        """
        Calculate substring match score.
        Returns ratio of query terms found in document.
        """
        query_norm = self._normalize_text(query)
        doc_norm = self._normalize_text(document)
        
        if query_norm in doc_norm:
            # Score based on relative length of match
            return len(query_norm) / len(doc_norm) if len(doc_norm) > 0 else 0.0
        
        # Check for partial word matches
        query_words = query_norm.split()
        doc_words = doc_norm.split()
        
        matches = sum(1 for word in query_words if word in doc_words)
        return matches / len(query_words) if len(query_words) > 0 else 0.0
    
    def _fuzzy_match_score(self, query: str, document: str) -> float:
        """
        Calculate fuzzy match score using sequence similarity.
        Returns similarity ratio between 0 and 1.
        """
        query_norm = self._normalize_text(query)
        doc_norm = self._normalize_text(document)
        
        return SequenceMatcher(None, query_norm, doc_norm).ratio()
    
    def _extract_document_text(self, document: dict, c_type: str = "text") -> str:
        """Extract text from document based on corpus type."""
        if isinstance(document, str):
            return document
        
        if c_type == "text":
            return document.get("text", "")
    
    def search(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               top_k: int,
               q_type: str = "text",
               c_type: str = "text",
               return_sorted: bool = True,
               **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Search for exact matches in the corpus.
        
        Args:
            corpus: Dictionary of corpus documents {doc_id: {title, text, ...}}
            queries: Dictionary of queries {query_id: query_text}
            top_k: Number of top results to return
            score_function: Not used (kept for interface compatibility)
            q_type: Query type (not used in exact match)
            c_type: Corpus type
            return_sorted: Whether to return results sorted by score
            
        Returns:
            Dictionary of results {query_id: {doc_id: score}}
        """
        logger.info(f"Starting exact match search with match_type: {self.match_type}")
        
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        
        # Use heap to maintain top-k results for each query
        result_heaps = {qid: [] for qid in query_ids}
        
        corpus_ids = list(corpus.keys())
        total_docs = len(corpus_ids)
        
        logger.info(f"Searching {len(queries)} queries against {total_docs} documents...")
        
        # Get score function
        score_func = self.score_functions[self.match_type]
        
        for doc_idx, corpus_id in enumerate(corpus_ids):
            if self.show_progress_bar and doc_idx % 1000 == 0:
                logger.info(f"Processing document {doc_idx + 1}/{total_docs}")
            
            # Extract document text
            document = corpus[corpus_id]
            doc_text = self._extract_document_text(document, c_type)
            
            # Score against all queries
            for query_id in query_ids:
                query_text = queries[query_id][q_type]
                
                # Skip self-matching if query_id == corpus_id
                if query_id == corpus_id:
                    continue
                
                # Calculate match score
                score = score_func(query_text, doc_text)
                
                # Only consider non-zero scores
                if score > 0:
                    if len(result_heaps[query_id]) < top_k:
                        # Push item on the heap (min-heap, so negate score)
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        # If score is better than the worst in heap, replace it
                        if score > result_heaps[query_id][0][0]:
                            heapq.heapreplace(result_heaps[query_id], (score, corpus_id))
        
        # Convert heaps to final results
        for qid in result_heaps:
            # Sort results by score (descending) if requested
            if return_sorted:
                sorted_results = sorted(result_heaps[qid], key=lambda x: x[0], reverse=True)
                for score, corpus_id in sorted_results:
                    self.results[qid][corpus_id] = score
            else:
                for score, corpus_id in result_heaps[qid]:
                    self.results[qid][corpus_id] = score
        
        logger.info(f"Search completed. Found matches for {len([qid for qid in self.results if self.results[qid]])} queries.")
        
        return self.results
