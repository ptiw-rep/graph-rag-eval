from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

def extract_ollama_facts(text: str, model="llama3") -> list:
    llm = Ollama(model=model)

    prompt = PromptTemplate.from_template(
        "Extract all subject-predicate-object triples from the following text:\n"
        "{text}\n"
        "Format: (subject, predicate, object)\n"
        "Triples:"
    )

    response = llm.invoke(prompt.format(text=text))

    facts = []
    for line in response.strip().split("\n"):
        if line.startswith("(") and line.endswith(")"):
            try:
                subj, pred, obj = eval(line)
                facts.append((subj.strip(), pred.strip(), obj.strip()))
            except:
                continue
    return facts