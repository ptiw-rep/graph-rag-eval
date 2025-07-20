from .spacy_ie import extract_spacy_facts, extract_openie_facts
from .ollama_llm import extract_ollama_facts

def hybrid_extract(text: str, model="llama3") -> list:
    spacy_facts = extract_spacy_facts(text)
    openie_facts = extract_openie_facts(text)
    ollama_facts = extract_ollama_facts(text, model=model)
    return list(set(spacy_facts + openie_facts + ollama_facts))