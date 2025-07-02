import json
import csv
from collections import defaultdict
from typing import List, Dict, Any

def extract_and_save_reranker_predictions(dataset, rerank_results: dict, k_values: List[int], output_file: str, split_num: int):
    """
    Extract prediction data from reranker results and save to file
    
    Args:
        dataset: Dataset containing queries and corpus with labels
        rerank_results: Dictionary with k as key and query results as values
        k_values: List of k values to process
        output_file: Base output file path
        split_num: Current CV split number
    """
    # Dictionary to store predictions for each k value
    all_predictions = {}
    
    for top_k in k_values:
        predictions_for_k = []
        
        for query_id in rerank_results[top_k]:
            true_label = dataset.queries[query_id]['labels']
            
            # Sort results by score in descending order
            sorted_results = sorted(rerank_results[top_k][query_id].items(), 
                                  key=lambda item: item[1], reverse=True)
            
            # Get top-k doc_ids for current k value
            top_k_doc_ids = [doc_id for doc_id, _ in sorted_results[:top_k]]
            
            # Count label frequencies in top-k results
            label_counts = defaultdict(int)
            for doc_id in top_k_doc_ids:
                predicted_label = dataset.corpus[doc_id]['labels']
                label_counts[predicted_label] += 1
            
            # Determine the most frequent label in top-k results
            predicted_label = None
            if label_counts:
                predicted_label = max(label_counts.items(), key=lambda x: x[1])[0]
            
            # Store prediction data
            prediction_data = {
                'query_id': query_id,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'is_correct': predicted_label == true_label if predicted_label is not None else False,
                'k': top_k,
                'split': split_num,
                'top_k_doc_ids': top_k_doc_ids,
                'label_counts': dict(label_counts)
            }
            predictions_for_k.append(prediction_data)
        
        all_predictions[top_k] = predictions_for_k
    
    # Save predictions
    save_reranker_predictions(all_predictions, output_file)
    
    return all_predictions

def save_reranker_predictions(predictions: Dict[int, List[Dict[str, Any]]], output_file: str):
    """
    Save reranker predictions to CSV and JSON files
    
    Args:
        predictions: Dictionary with k values as keys and list of prediction data as values
        output_file: Base output file path (extensions will be added)
    """
    # Save as CSV
    csv_file = output_file.replace('.csv', '') + '.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['query_id', 'true_label', 'predicted_label', 'is_correct', 'k', 'split', 'top_k_doc_ids', 'label_counts'])
        
        # Write data for all k values
        for k_val, pred_list in predictions.items():
            for pred in pred_list:
                writer.writerow([
                    pred['query_id'],
                    pred['true_label'],
                    pred['predicted_label'],
                    pred['is_correct'],
                    pred['k'],
                    pred['split'],
                    '|'.join(pred['top_k_doc_ids']),  # Join doc IDs with separator
                    json.dumps(pred['label_counts'])  # Convert dict to JSON string
                ])
    
    # Save as JSON for easier programmatic access
    json_file = output_file.replace('.csv', '') + '.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"Reranker predictions saved to {csv_file} and {json_file}")
