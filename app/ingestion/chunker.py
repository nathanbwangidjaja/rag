"""Chunk text at sentence boundaries for embedding."""

import re


def split_sentences(text: str) -> list[str]:
    # split on period/question/exclamation followed by space or newline
    parts = re.split(r'(?<=[.!?])\s+', text)
    # also split on double newlines (paragraph breaks)
    result = []
    for part in parts:
        if "\n\n" in part:
            sub = part.split("\n\n")
            result.extend(s.strip() for s in sub if s.strip())
        else:
            if part.strip():
                result.append(part.strip())
    return result


def chunk_text(text: str, max_chars=512, overlap=64) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        # if a single sentence exceeds max, just take it as its own chunk
        if sent_len > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            chunks.append(sent)
            continue

        if current_len + sent_len + 1 > max_chars and current:
            chunk_text_joined = " ".join(current)
            chunks.append(chunk_text_joined)

            # figure out overlap: keep trailing sentences that fit within overlap chars
            overlap_sents = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) + 1 > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s) + 1

            current = overlap_sents
            current_len = overlap_len

        current.append(sent)
        current_len += sent_len + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_pages(pages: list[dict], source: str, max_chars=512, overlap=64) -> list[dict]:
    all_chunks = []
    idx = 0

    for page_info in pages:
        page_chunks = chunk_text(page_info["text"], max_chars=max_chars, overlap=overlap)
        for chunk in page_chunks:
            all_chunks.append({
                "id": f"{source}::chunk_{idx}",
                "text": chunk,
                "source": source,
                "page": page_info["page"],
                "index": idx,
            })
            idx += 1

    return all_chunks
