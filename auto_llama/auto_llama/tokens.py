import tiktoken


def count_tokens(input: str):
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(input))
