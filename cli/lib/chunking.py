import re


def semantic_chunking(text: str, overlap: int = 0, max_chunk_size: int = 4) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not sentences[0].strip().endswith((".", "!", "?")):
        sentences = [text]

    chunks = []
    step_size = max_chunk_size - overlap
    sentences = [s.strip() for s in sentences if s.strip() ]
    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i:i+max_chunk_size]
        if len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
    return chunks


def fixed_sized_chunking(text: str, overlap, chunk_size: int = 200) -> list[str]:
    words = text.split()
    chunks = []
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
        chunk_words = words[i:i+chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
    return chunks
