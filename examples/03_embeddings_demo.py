from __future__ import annotations

from l3store.embeddings import CachedEmbeddingService, SentenceTransformerService


def main() -> None:
    service = SentenceTransformerService(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        normalize=True,
    )
    embeddings = CachedEmbeddingService(service, max_cache_size=10_000)

    texts = [
        "Quantum computing uses qubits.",
        "Qubits are the basic units of quantum computers.",
        "A chocolate cake needs flour and cocoa.",
    ]

    vectors = embeddings.embed_batch(texts)
    print(f"Embedding dimension: {embeddings.dimension()}")
    print(f"Batch shape: {vectors.shape}")
    print(
        "Quantum similarity:",
        f"{embeddings.similarity(vectors[0], vectors[1]):.3f}",
    )
    print(
        "Different topic similarity:",
        f"{embeddings.similarity(vectors[0], vectors[2]):.3f}",
    )
    print(f"Cache stats: {embeddings.cache_stats()}")


if __name__ == "__main__":
    main()
