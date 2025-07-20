from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def fuzzy_match_relation(rel1: str, rel2: str, threshold=0.75) -> bool:
    emb1 = model.encode(rel1, convert_to_tensor=True)
    emb2 = model.encode(rel2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return score >= threshold


def compare_graphs(source_graph, summary_graph, threshold=0.75):
    matched = []
    unmatched = []

    for u, v, data in source_graph.edges(data=True):
        rel1 = data['label']
        if summary_graph.has_edge(u, v):
            rel2 = summary_graph[u][v]['label']
            if fuzzy_match_relation(rel1, rel2, threshold):
                matched.append((u, rel1, v))
            else:
                unmatched.append((u, rel1, v, f"Summary: {rel2} (score: {util.cos_sim(model.encode(rel1), model.encode(rel2)).item():.2f})"))
        else:
            unmatched.append((u, rel1, v, "No edge in summary"))

    return matched, unmatched