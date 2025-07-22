# graph_comparer/embedding_comparer.py

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def fuzzy_match_relation(rel1: str, rel2: str, threshold=0.75) -> bool:
    emb1 = model.encode(rel1, convert_to_tensor=True)
    emb2 = model.encode(rel2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return score >= threshold


def compare_graphs(source_graph, summary_graph, threshold=0.75):
    """
    Compare source and summary graphs for recall and precision.
    
    Returns:
        recall_results: {matched_relations: [], partial: [], missing: []}
        precision_results: {correct_relations: [], partial: [], hallucinations: []}
    """
    # --- Recall: Source → Summary ---
    recall_matched_relations = []
    recall_partial = []
    recall_missing = []

    for u, v, data in source_graph.edges(data=True):
        rel1 = data['label']
        if summary_graph.has_edge(u, v):
            rel2 = summary_graph[u][v]['label']
            if fuzzy_match_relation(rel1, rel2, threshold):
                recall_matched_relations.append((u, rel1, v))
            else:
                recall_partial.append((u, rel1, v, rel2))
        else:
            recall_missing.append((u, rel1, v))

    # --- Precision: Summary → Source ---
    precision_correct_relations = []
    precision_partial = []
    precision_hallucinations = []

    for u, v, data in summary_graph.edges(data=True):
        rel2 = data['label']
        if source_graph.has_edge(u, v):
            rel1 = source_graph[u][v]['label']
            if fuzzy_match_relation(rel1, rel2, threshold):
                precision_correct_relations.append((u, rel2, v))
            else:
                precision_partial.append((u, rel2, v, rel1))
        else:
            precision_hallucinations.append((u, rel2, v))

    return {
        "recall": {
            "matched_relations": recall_matched_relations,
            "partial": recall_partial,
            "missing": recall_missing
        },
        "precision": {
            "correct_relations": precision_correct_relations,
            "partial": precision_partial,
            "hallucinations": precision_hallucinations
        }
    }