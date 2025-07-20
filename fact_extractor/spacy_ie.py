import spacy
from openie import StanfordOpenIE

def extract_spacy_facts(text: str, model) -> list:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    triples = []
    for token in doc:
        if "subj" in token.dep_:
            subject = token.text
            verb = token.head.text
            for child in token.head.children:
                if "obj" in child.dep_:
                    triples.append((subject, verb, child.text))
    return triples


def extract_openie_facts(text: str) -> list:
    with StanfordOpenIE() as client:
        return client.annotate(text)