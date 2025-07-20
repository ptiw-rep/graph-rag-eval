from py2neo import Graph, Node, Relationship

def build_neo4j_graph(facts: list, uri="bolt://localhost:7687", user="neo4j", password="your_password"):
    neo_graph = Graph(uri, auth=(user, password))
    neo_graph.delete_all()

    nodes = {}
    for subj, rel, obj in facts:
        if subj not in nodes:
            nodes[subj] = Node("Entity", name=subj)
        if obj not in nodes:
            nodes[obj] = Node("Entity", name=obj)
        neo_graph.create(Relationship(nodes[subj], rel, nodes[obj]))

    return neo_graph