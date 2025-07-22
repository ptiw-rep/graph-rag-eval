# cli/cli.py

import click
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import modules
from chunker.text_chunker import chunk_text
from fact_extractor.hybrid_extractor import hybrid_extract
from graph_builder.networkx_builder import build_graph
from graph_comparer.embedding_comparer import compare_graphs
from visualizer.nx_visualizer import visualize_graph
from visualizer.neo4j_visualizer import visualize_neo4j_graph
from graph_builder.neo4j_builder import build_neo4j_graph
from metrics.scorer import calculate_recall_score, calculate_precision_score, normalize_score
from utils.config_loader import load_config

@click.command()
@click.option('--source', prompt='Enter source text', help='Original text')
@click.option('--summary', prompt='Enter summary text', help='Summarized text')
@click.option('--method', type=click.Choice(['spacy', 'ollama', 'hybrid']), default='hybrid')
@click.option('--model', default=None, help='Model name for Ollama (overrides config)')
@click.option('--threshold', default=None, type=float, help='Cosine similarity threshold (overrides config)')
@click.option('--chunk/--no-chunk', default=None, help='Enable chunking (overrides config)')
@click.option('--store/--no-store', default=True, help='Store in Neo4j')
def evaluate(source, summary, method, model, threshold, chunk, store):

    # Load config
    try:
        config = load_config()
    except Exception as e:
        click.echo(f"❌ Error loading config: {e}")
        return

    # Override with CLI args or use config
    model = model or config['ollama_model']
    threshold = threshold or config['similarity_threshold']
    chunk = chunk if chunk is not None else True  # Default to True unless specified
    chunk_size = config.get('chunk_size', 500)

    recall_weights = config['weights']['recall']
    precision_weights = config['weights']['precision']
    export_config = config['export']
    export_path = Path(export_config['path'])
    export_path.mkdir(exist_ok=True)  # Create results dir

    # Chunking
    if chunk:
        source_chunks = chunk_text(source, chunk_size=chunk_size)
        summary_chunks = chunk_text(summary, chunk_size=chunk_size)
    else:
        source_chunks = [source]
        summary_chunks = [summary]

    # Extract facts
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
        neo4j_cfg = config['neo4j']
        build_neo4j_graph(source_facts, uri=neo4j_cfg['uri'], user=neo4j_cfg['user'], password=neo4j_cfg['password'])
        build_neo4j_graph(summary_facts, uri=neo4j_cfg['uri'], user=neo4j_cfg['user'], password=neo4j_cfg['password'])
        visualize_neo4j_graph(uri=neo4j_cfg['uri'], user=neo4j_cfg['user'], password=neo4j_cfg['password'])
        click.echo("✅ Graphs stored and visualized in Neo4j")

    # Compare
    comparison = compare_graphs(source_graph, summary_graph, threshold=threshold)

    # Compute Scores
    recall_res = comparison["recall"]
    prec_res = comparison["precision"]

    total_source = len(recall_res["matched_relations"]) + len(recall_res["partial"]) + len(recall_res["missing"])
    total_summary = len(prec_res["correct_relations"]) + len(prec_res["partial"]) + len(prec_res["hallucinations"])

    recall_score, recall_max = calculate_recall_score(
        recall_res["matched_relations"],
        recall_res["partial"],
        total_source,
        recall_weights
    )

    precision_score, precision_max = calculate_precision_score(
        prec_res["correct_relations"],
        prec_res["partial"],
        prec_res["hallucinations"],
        precision_weights
    )

    norm_recall = normalize_score(recall_score, recall_max)
    norm_precision = normalize_score(precision_score, precision_max)
    f1_score = 0.0
    if norm_recall + norm_precision > 0:
        f1_score = 2 * (norm_recall * norm_precision) / (norm_recall + norm_precision)

    # Generate unique ID for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"eval_{timestamp}"

    # Prepare result dict
    result = {
        "run_id": run_id,
        "timestamp": timestamp,
        "source_text_preview": source[:200] + "..." if len(source) > 200 else source,
        "summary_text_preview": summary[:200] + "..." if len(summary) > 200 else summary,
        "extraction_method": method,
        "ollama_model": model,
        "similarity_threshold": threshold,
        "fact_counts": {
            "source_total": total_source,
            "summary_total": total_summary,
            "recall": {
                "matched_relations": len(recall_res["matched_relations"]),
                "partial_relation": len(recall_res["partial"]),
                "missing": len(recall_res["missing"])
            },
            "precision": {
                "correct_relations": len(prec_res["correct_relations"]),
                "wrong_relation": len(prec_res["partial"]),
                "hallucinated": len(prec_res["hallucinations"])
            }
        },
        "raw_facts": {
            "source": [list(fact) for fact in source_facts],
            "summary": [list(fact) for fact in summary_facts]
        },
        "unmatched": {
            "missed_in_summary": [list(fact) for fact in recall_res["missing"]],
            "hallucinated_in_summary": [list(fact) for fact in prec_res["hallucinations"]]
        },
        "scores": {
            "recall": {
                "score": recall_score,
                "max": recall_max,
                "normalized": norm_recall
            },
            "precision": {
                "score": precision_score,
                "max": precision_max,
                "normalized": norm_precision
            },
            "f1_normalized": f1_score
        }
    }

    # Export Results
    if export_config['json']:
        json_path = export_path / f"{run_id}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        click.echo(f"📄 JSON report saved: {json_path}")

    if export_config['csv']:
        # Flatten some parts for CSV
        df_data = []
        for fact in recall_res["matched_relations"]:
            df_data.append({"type": "recall_match", "subject": fact[0], "relation": fact[1], "object": fact[2]})
        for fact in recall_res["partial"]:
            df_data.append({"type": "recall_partial", "subject": fact[0], "relation": fact[1], "object": fact[2], "in_summary_as": fact[3]})
        for fact in recall_res["missing"]:
            df_data.append({"type": "recall_missing", "subject": fact[0], "relation": fact[1], "object": fact[2]})

        for fact in prec_res["correct_relations"]:
            df_data.append({"type": "precision_match", "subject": fact[0], "relation": fact[1], "object": fact[2]})
        for fact in prec_res["partial"]:
            df_data.append({"type": "precision_wrong_rel", "subject": fact[0], "relation": fact[1], "object": fact[2], "in_source_as": fact[3]})
        for fact in prec_res["hallucinations"]:
            df_data.append({"type": "precision_hallucination", "subject": fact[0], "relation": fact[1], "object": fact[2]})

        csv_path = export_path / f"{run_id}.csv"
        pd.DataFrame(df_data).to_csv(csv_path, index=False, encoding='utf-8')
        click.echo(f"📊 CSV report saved: {csv_path}")

    # Print Summary
    click.echo("\n" + "="*60)
    click.echo("📊 EVALUATION RESULTS")
    click.echo("="*60)
    click.echo(f"🔍 Recall Score:       {recall_score:.2f} / {recall_max:.2f} → Normalized: {norm_recall:.3f}")
    click.echo(f"🎯 Precision Score:    {precision_score:.2f} / {precision_max:.2f} → Normalized: {norm_precision:.3f}")
    click.echo(f"⭐ F1 Score:           {f1_score:.3f}")
    click.echo(f"📁 Exported to: {export_path}")

    if prec_res["hallucinations"]:
        click.echo("\n❌ Detected Hallucinations:")
        for fact in prec_res["hallucinations"]:
            click.echo(f"  • {fact}")


if __name__ == '__main__':
    evaluate()