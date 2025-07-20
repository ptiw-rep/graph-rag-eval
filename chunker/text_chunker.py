import textwrap

def chunk_text(text: str, chunk_size=500) -> list:
    return textwrap.wrap(text, chunk_size, break_long_words=False, replace_whitespace=False)