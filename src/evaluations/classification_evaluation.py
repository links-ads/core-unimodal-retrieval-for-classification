from __future__ import annotations

import logging
import json
import csv
from collections import defaultdict
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class EvaluateClassification:
    """
    Evaluate the classification task using accuracy, precision, recall, and F1 score
    with both macro and micro averaging
    """
    def __init__(self):
        pass
    
    def evaluate_classifier(   
                                self,
                                pred: list[str], 
                                true: list[str], 
                            ) -> dict[str, float]:
        """
        Evaluate the classification task using accuracy, precision, recall, and F1 score
        with macro averaging
        Args:
            pred: list of predicted labels
            true: list of true labels
        Returns:
            Dictionary with accuracy, precision, recall, and F1 score
        """
        
        metrics = {}
        f1 = f1_score(true, pred, average='macro')
        precision = precision_score(true, pred, average='macro')
        recall = recall_score(true, pred, average='macro')
        accuracy = accuracy_score(true, pred)
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro Precision: {precision:.4f}")
        logger.info(f"Macro Recall: {recall:.4f}")
        logger.info(f"Macro F1: {f1:.4f}")
        # Store metrics
        metrics = {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1
        }
        return metrics
    
    def evaluate_decoder(   
                            self,
                            dataset, 
                            results: dict[str, dict[str, bool]], 
                            k: List[int]
                        ) -> dict[str, dict[str, float]]:
        preds = []
        trues = []
        metrics_by_k = {}
        
        for topk in k:
            for query_id in results:
                pred_label = results[query_id][topk]
                # get the true label
                true_label = dataset.queries[query_id]['labels']
                preds.append(pred_label)
                trues.append(true_label)
            # Calculate f1-score
            f1 = f1_score(trues, preds, average='macro')
            precision = precision_score(trues, preds, average='macro')
            recall = recall_score(trues, preds, average='macro')
            accuracy = accuracy_score(trues, preds)
            logger.info(f"Accuracy@{topk}: {accuracy:.4f}")
            logger.info(f"Macro Precision@{topk}: {precision:.4f}")
            logger.info(f"Macro Recall@{topk}: {recall:.4f}")
            logger.info(f"Macro F1@{topk}: {f1:.4f}")
            # Store metrics for current k
            metrics_by_k[topk] = {
                                    "accuracy": accuracy,
                                    "macro_precision": precision,
                                    "macro_recall": recall,
                                    "macro_f1": f1
                                }
        return metrics_by_k
    
    def evaluate_reranker(
        self,
        dataset,
        results: dict[str, dict[str, float]],
        k: List[int],
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate the classification task using accuracy, precision, recall, and F1 score
        
        Args:
            dataset: Dataset containing queries and corpus with labels
            results: Dictionary with query_id as key and dict of doc_id -> score as value
            k: List of values for top-k evaluation
            
        Returns:
            Dictionary with k values as keys and evaluation metrics as values
            (each containing accuracy, precision, recall, f1 with both macro and micro averages)
        """
        # Dictionary to store metrics for each k
        metrics_by_k = {}
        
        for top_k in k:
            total = 0
            
            # For macro-average metrics
            label_true_pos = defaultdict(int)  # True positives per label
            label_pred_total = defaultdict(int)  # Total predictions per label
            label_true_total = defaultdict(int)  # Total actual instances per label
            
            # For micro-average metrics
            micro_true_pos = 0  # Total true positives across all labels
            
            for query_id in results[top_k]:
                true_label = dataset.queries[query_id]['labels']
                # Count true label instances
                label_true_total[true_label] += 1
                total += 1
                
                # Sort results by score in descending order
                sorted_results = sorted(results[top_k][query_id].items(), key=lambda item: item[1], reverse=True)
                
                # Get top-k doc_ids for current k value
                top_k_doc_ids = [doc_id for doc_id, _ in sorted_results[:top_k]]
                
                # Count label frequencies in top-k results
                label_counts = defaultdict(int)
                for doc_id in top_k_doc_ids:
                    predicted_label = dataset.corpus[doc_id]['labels']
                    label_counts[predicted_label] += 1
                
                # Determine the most frequent label in top-k results
                if label_counts:
                    predicted_label = max(label_counts.items(), key=lambda x: x[1])[0]
                    
                    # Update prediction counters
                    label_pred_total[predicted_label] += 1
                    
                    # Check if prediction is correct
                    if predicted_label == true_label:
                        label_true_pos[true_label] += 1
                        micro_true_pos += 1
            
            # Calculate accuracy
            accuracy = micro_true_pos / total if total > 0 else 0
            
            # Get all unique labels
            all_labels = set(label_true_total.keys()) | set(label_pred_total.keys())
            
            # Calculate macro-average metrics
            precisions = []
            recalls = []
            f1_scores = []
            
            for label in all_labels:
                # Precision: true positives / (true positives + false positives)
                precision = label_true_pos[label] / label_pred_total[label] if label_pred_total[label] > 0 else 0
                precisions.append(precision)
                
                # Recall: true positives / (true positives + false negatives)
                recall = label_true_pos[label] / label_true_total[label] if label_true_total[label] > 0 else 0
                recalls.append(recall)
                
                # F1 score: 2 * (precision * recall) / (precision + recall)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
            
            # Calculate macro-averaged metrics
            macro_precision = sum(precisions) / len(precisions) if precisions else 0
            macro_recall = sum(recalls) / len(recalls) if recalls else 0
            macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            
            # Calculate micro-averaged metrics
            micro_pred_total = sum(label_pred_total.values())  # Total number of predictions
            micro_true_total = total  # Total number of queries
            
            micro_precision = micro_true_pos / micro_pred_total if micro_pred_total > 0 else 0
            micro_recall = micro_true_pos / micro_true_total if micro_true_total > 0 else 0
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
            
            # Log results for current k
            logger.info(f"Accuracy@{top_k}: {accuracy:.4f}")
            logger.info(f"Macro Precision@{top_k}: {macro_precision:.4f}")
            logger.info(f"Macro Recall@{top_k}: {macro_recall:.4f}")
            logger.info(f"Macro F1@{top_k}: {macro_f1:.4f}")
            logger.info(f"Micro Precision@{top_k}: {micro_precision:.4f}")
            logger.info(f"Micro Recall@{top_k}: {micro_recall:.4f}")
            logger.info(f"Micro F1@{top_k}: {micro_f1:.4f}")
            
            # Store metrics for current k
            metrics_by_k[top_k] = {
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1
            }
        
        # Return metrics for all k values
        return metrics_by_k

    def _get_majority_vote_prediction(self, dataset, top_k_doc_ids):
        """
        Get prediction using majority voting (original approach)
        """
        # Count label frequencies in top-k results
        label_counts = defaultdict(int)
        for doc_id in top_k_doc_ids:
            predicted_label = dataset.corpus[doc_id]['labels']
            label_counts[predicted_label] += 1
        
        # Determine the most frequent label in top-k results
        if label_counts:
            return max(label_counts.items(), key=lambda x: x[1])[0]
        return None

    def _get_weighted_prediction(self, dataset, top_k_results):
        """
        Get prediction using weighted average based on similarity scores
        
        Args:
            dataset: Dataset containing corpus with labels
            top_k_results: List of tuples (doc_id, similarity_score)
        
        Returns:
            Predicted label based on weighted scores
        """
        if not top_k_results:
            return None
        
        # Calculate weighted scores for each label
        label_weighted_scores = defaultdict(float)
        total_weight = 0
        
        for doc_id, score in top_k_results:
            label = dataset.corpus[doc_id]['labels']
            # Use the similarity score as weight
            weight = score
            label_weighted_scores[label] += weight
            total_weight += weight
        
        # Normalize the weighted scores (optional, but helps with interpretation)
        if total_weight > 0:
            for label in label_weighted_scores:
                label_weighted_scores[label] /= total_weight
        
        # Return the label with the highest weighted score
        if label_weighted_scores:
            return max(label_weighted_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def evaluate_retriever(
        self,
        dataset,
        results: dict[str, dict[str, float]],
        k: List[int],
        use_weighted_scoring: bool = False,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate the classification task using accuracy, precision, recall, and F1 score
        
        Args:
            dataset: Dataset containing queries and corpus with labels
            results: Dictionary with query_id as key and dict of doc_id -> score as value
            k: List of values for top-k evaluation
            use_weighted_scoring: If True, use weighted average based on similarity scores.
                                If False, use majority voting (default behavior)
            
        Returns:
            Dictionary with k values as keys and evaluation metrics as values
            (each containing accuracy, precision, recall, f1 with both macro and micro averages)
        """
        # Dictionary to store metrics for each k
        metrics_by_k = {}
        
        for top_k in k:
            total = 0
            
            # For macro-average metrics
            label_true_pos = defaultdict(int)  # True positives per label
            label_pred_total = defaultdict(int)  # Total predictions per label
            label_true_total = defaultdict(int)  # Total actual instances per label
            
            # For micro-average metrics
            micro_true_pos = 0  # Total true positives across all labels
            
            for query_id in results:
                true_label = dataset.queries[query_id]['labels']
                # Count true label instances
                label_true_total[true_label] += 1
                total += 1
                
                # Sort results by score in descending order
                sorted_results = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)
                
                # Get top-k doc_ids for current k value
                top_k_results = sorted_results[:top_k]
                
                if use_weighted_scoring:
                    # Weighted scoring approach
                    predicted_label = self._get_weighted_prediction(dataset, top_k_results)
                else:
                    # Majority voting approach (original behavior)
                    top_k_doc_ids = [doc_id for doc_id, _ in top_k_results]
                    predicted_label = self._get_majority_vote_prediction(dataset, top_k_doc_ids)
                
                # Update prediction counters
                if predicted_label is not None:
                    label_pred_total[predicted_label] += 1
                    
                    # Check if prediction is correct
                    if predicted_label == true_label:
                        label_true_pos[true_label] += 1
                        micro_true_pos += 1
            
            # Calculate accuracy
            accuracy = micro_true_pos / total if total > 0 else 0
            
            # Get all unique labels
            all_labels = set(label_true_total.keys()) | set(label_pred_total.keys())
            
            # Calculate macro-average metrics
            precisions = []
            recalls = []
            f1_scores = []
            
            for label in all_labels:
                # Precision: true positives / (true positives + false positives)
                precision = label_true_pos[label] / label_pred_total[label] if label_pred_total[label] > 0 else 0
                precisions.append(precision)
                
                # Recall: true positives / (true positives + false negatives)
                recall = label_true_pos[label] / label_true_total[label] if label_true_total[label] > 0 else 0
                recalls.append(recall)
                
                # F1 score: 2 * (precision * recall) / (precision + recall)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
            
            # Calculate macro-averaged metrics
            macro_precision = sum(precisions) / len(precisions) if precisions else 0
            macro_recall = sum(recalls) / len(recalls) if recalls else 0
            macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            
            # Calculate micro-averaged metrics
            micro_pred_total = sum(label_pred_total.values())  # Total number of predictions
            micro_true_total = total  # Total number of queries
            
            micro_precision = micro_true_pos / micro_pred_total if micro_pred_total > 0 else 0
            micro_recall = micro_true_pos / micro_true_total if micro_true_total > 0 else 0
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
            
            # Log results for current k
            scoring_method = "Weighted" if use_weighted_scoring else "Majority Vote"
            logger.info(f"[{scoring_method}] Accuracy@{top_k}: {accuracy:.4f}")
            logger.info(f"[{scoring_method}] Macro Precision@{top_k}: {macro_precision:.4f}")
            logger.info(f"[{scoring_method}] Macro Recall@{top_k}: {macro_recall:.4f}")
            logger.info(f"[{scoring_method}] Macro F1@{top_k}: {macro_f1:.4f}")
            logger.info(f"[{scoring_method}] Micro Precision@{top_k}: {micro_precision:.4f}")
            logger.info(f"[{scoring_method}] Micro Recall@{top_k}: {micro_recall:.4f}")
            logger.info(f"[{scoring_method}] Micro F1@{top_k}: {micro_f1:.4f}")
            
            # Store metrics for current k
            metrics_by_k[top_k] = {
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1
            }
        
        # Return metrics for all k values
        return metrics_by_k