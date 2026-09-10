"""Compute pair-specific similarity using the configured production embedding backend."""
import math

from pydantic import BaseModel, ConfigDict, Field


class SimilarityPair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pairId: str = Field(min_length=1, max_length=500)
    issueText: str = Field(min_length=1, max_length=4000)
    recordText: str = Field(min_length=1, max_length=4000)


class SimilarityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pairs: list[SimilarityPair] = Field(min_length=1, max_length=64)


def score_pairs(request: SimilarityRequest, client) -> dict:
    if len({p.pairId for p in request.pairs}) != len(request.pairs):
        raise ValueError("Duplicate pairId")
    texts = list(dict.fromkeys(t for p in request.pairs for t in (p.issueText, p.recordText)))
    embeddings = client.embed(texts)
    if len(embeddings) != len(texts):
        raise ValueError("Incomplete embedding response")
    vectors = {}
    dimension = None
    for text, embedding in zip(texts, embeddings, strict=True):
        vector = embedding.dense
        if not vector or not all(math.isfinite(x) for x in vector):
            raise ValueError("Invalid embedding")
        dimension = dimension or len(vector)
        norm = math.sqrt(sum(x * x for x in vector))
        if len(vector) != dimension or not math.isfinite(norm) or norm == 0:
            raise ValueError("Invalid embedding dimension or norm")
        vectors[text] = [x / norm for x in vector]
    return {"scores": [{"pairId": p.pairId, "score": max(0.0, min(1.0, sum(
        a * b for a, b in zip(vectors[p.issueText], vectors[p.recordText], strict=True)
    )))} for p in request.pairs]}
