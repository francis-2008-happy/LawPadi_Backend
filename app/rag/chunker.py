import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_text(text, min_tokens=500, max_tokens=800):
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        piece = tokens[i : i + max_tokens]
        if len(piece) >= min_tokens:
            chunks.append(tokenizer.decode(piece))

    return chunks
