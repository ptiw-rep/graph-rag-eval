from py2neo import Graph
import matplotlib.pyplot as plt
import networkx as nx

def visualize_neo4j_graph(uri="bolt://localhost:7687", user="neo4j", password="your_password"):
    neo_graph = Graph(uri, auth=(user, password))
    G = nx.DiGraph()

    query = """
    MATCH (s)-[r]->(o)
    RETURN s.name AS subject, type(r) AS relation, o.name AS object
    """
    results = neo_graph.run(query).data()

    for record in results:
        G.add_edge(record['subject'], record['object'], label=record['relation'])

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightgreen', font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Neo4j Knowledge Graph")
    plt.show()