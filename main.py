import json
from cli.cli import evaluate

def run_sample(sample_name="sample1"):
    with open("sample_inputs/test_samples.json") as f:
        samples = json.load(f)

    sample = samples[sample_name]
    source = sample["source"]
    summary = sample["summary"]

    # Run CLI logic manually
    print("Running evaluation with sample:", sample_name)
    evaluate.callback(source, summary, 'spacy', 'gemma3:4b', 0.75, True, False)

if __name__ == "__main__":
    run_sample("sample2")