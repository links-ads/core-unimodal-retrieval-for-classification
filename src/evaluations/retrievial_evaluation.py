from __future__ import annotations
import logging
import pytrec_eval
import logging
import json
import csv
from collections import defaultdict
from typing import List, Dict, Any

from modules.retriever.search.dense.search import DenseRetrievalExactSearch

logger = logging.getLogger(__name__)


class EvaluateRetrieval:
    def __init__(
        self,
        retriever: DenseRetrievalExactSearch = None,
        k_values: list[int] = [1, 3, 5, 10, 100, 1000],
        score_function: str = "dot",
    ):
        self.k_values = k_values
        self.top_k = max(k_values)
        self.retriever = retriever
        self.score_function = score_function

    def retrieve(
        self, corpus: dict[str, dict[str, str]], queries: dict[str, str], **kwargs
    ) -> dict[str, dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")
        return self.retriever.search(corpus, queries, self.top_k, self.score_function, **kwargs)

    # def rerank(
    #     self,
    #     corpus: dict[str, dict[str, str]],
    #     queries: dict[str, str],
    #     results: dict[str, dict[str, float]],
    #     top_k: int,
    # ) -> dict[str, dict[str, float]]:
    #     new_corpus = {}

    #     for query_id in results:
    #         if len(results[query_id]) > top_k:
    #             for doc_id, _ in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
    #                 new_corpus[doc_id] = corpus[doc_id]
    #         else:
    #             for doc_id in results[query_id]:
    #                 new_corpus[doc_id] = corpus[doc_id]

    #     return self.retriever.search(new_corpus, queries, top_k, self.score_function)

    @staticmethod
    def evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        ignore_identical_ids: bool = True,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        if ignore_identical_ids:
            logger.info(
                "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."
            )
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        for eval in [ndcg, _map, recall, precision]:
            logger.info("\n")
            for k in eval.keys():
                logger.info(f"{k}: {eval[k]:.4f}")

        return ndcg, _map, recall, precision