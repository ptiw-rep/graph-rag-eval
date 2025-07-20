import click
from chunker.text_chunker import chunk_text
from fact_extractor.hybrid_extractor import hybrid_extract
from graph_builder.networkx_builder import build_graph
from graph_comparer.embedding_comparer import compare_graphs
from visualizer.nx_visualizer import visualize_graph
from visualizer.neo4j_visualizer import visualize_neo4j_graph
from graph_builder.neo4j_builder import build_neo4j_graph

@click.command()
@click.option('--source', prompt='Enter source text', help='Original text')
@click.option('--summary', prompt='Enter summary text', help='Summarized text')
@click.option('--method', type=click.Choice(['spacy', 'ollama', 'hybrid']), default='hybrid')
@click.option('--model', default='llama3', help='Model name for Ollama')
@click.option('--threshold', default=0.75, help='Cosine similarity threshold for fuzzy matching')
@click.option('--chunk/--no-chunk', default=True, help='Enable chunking for long texts')
@click.option('--store/--no-store', default=True, help='Store graph in Neo4j')
def evaluate(source, summary, method, model, threshold, chunk, store):

    # Chunking
    if chunk:
        source_chunks = chunk_text(source)
        summary_chunks = chunk_text(summary)
    else:
        source_chunks = [source]
        summary_chunks = [summary]

    # Extract facts per chunk
    source_facts = []
    summary_facts = []

    for text in source_chunks:
        if method == 'spacy':
            from fact_extractor.spacy_ie import extract_spacy_facts as extract
        elif method == 'ollama':
            from fact_extractor.ollama_llm import extract_ollama_facts as extract
        else:
            from fact_extractor.hybrid_extractor import hybrid_extract as extract
        source_facts.extend(extract(text, model=model))

    for text in summary_chunks:
        summary_facts.extend(extract(text, model=model))

    # Build graphs
    source_graph = build_graph(source_facts)
    summary_graph = build_graph(summary_facts)

    # Visualize
    visualize_graph(source_graph, "Source Graph")
    visualize_graph(summary_graph, "Summary Graph")

    # Store in Neo4j
    if store:
        build_neo4j_graph(source_facts)
        build_neo4j_graph(summary_facts)
        visualize_neo4j_graph()
        click.echo("Graphs stored and visualized in Neo4j")

    # Compare
    matched, unmatched = compare_graphs(source_graph, summary_graph, threshold=threshold)

    print("\n--- Matched Facts ---")
    for fact in matched:
        print(f"  {fact}")

    print("\n--- Unmatched / Mismatched Facts ---")
    for fact in unmatched:
        print(f"  {fact}")

if __name__ == '__main__':
    evaluate()