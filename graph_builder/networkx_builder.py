import networkx as nx

def build_graph(facts: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for subj, rel, obj in facts:
        G.add_edge(subj, obj, label=rel)
    return G