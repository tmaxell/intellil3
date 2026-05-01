"""
Минимальный пример работы с L3-хранилищем.

    docker compose -f deploy/docker/docker-compose.yml up -d
    pip install -e "."
    python examples/01_quickstart.py
"""

import numpy as np

from l3store.core.object_store import UnifiedObjectStore
from l3store.core.types import KVCacheBlock, RAGObject, SemanticCacheEntry
from l3store.storage.s3_backend import S3Backend
from l3store.utils.config import L3Config


def main():
    config = L3Config.from_yaml("configs/default.yaml")
    backend = S3Backend(
        endpoint_url=config.storage.endpoint_url,
        access_key=config.storage.access_key,
        secret_key=config.storage.secret_key,
        bucket=config.storage.bucket,
        region=config.storage.region,
    )
    store = UnifiedObjectStore(backend=backend, config=config)

    # KV-cache block
    print("=== KV-Cache Block ===")
    num_layers, block_size, num_heads, head_dim = 32, 16, 8, 128
    key_states = np.random.randn(num_layers, block_size, num_heads, head_dim).astype(np.float16)
    value_states = np.random.randn(num_layers, block_size, num_heads, head_dim).astype(np.float16)

    block = KVCacheBlock(
        model_name="meta-llama/Llama-3-8B",
        token_ids=list(range(100, 116)),
        block_index=0,
        shared_by_sessions=["user_alice_session_1"],
    )
    obj_id = store.put_kv_block(block, key_states, value_states)
    print(f"  Stored: {obj_id} ({key_states.nbytes + value_states.nbytes:,} bytes)")

    loaded_block, loaded_keys, loaded_values = store.get_kv_block(obj_id)
    assert np.array_equal(loaded_keys, key_states)
    print(f"  Loaded: token_hash={loaded_block.token_hash}, tensors match ✓")

    # RAG object
    print("\n=== RAG Object ===")
    embedding = np.random.randn(384).astype(np.float32)
    rag_obj = RAGObject(
        document_id="doc_quantum_001",
        chunk_index=3,
        chunk_text="Quantum entanglement allows particles to be correlated...",
        source="arxiv:2401.12345",
    )
    rag_id = store.put_rag_object(rag_obj, embedding)
    loaded_rag, _ = store.get_rag_object(rag_id)
    print(f"  Stored: {rag_id} (doc={loaded_rag.document_id}, dim={loaded_rag.embedding_dim})")

    # Semantic cache
    print("\n=== Semantic Cache ===")
    prompt_emb = np.random.randn(768).astype(np.float32)
    entry = SemanticCacheEntry(
        prompt_text="Explain quantum computing in simple terms",
        response_text="Quantum computing uses quantum bits (qubits)...",
        model_name="meta-llama/Llama-3-8B",
    )
    sem_id = store.put_semantic_entry(entry, prompt_emb)
    loaded_entry, _ = store.get_semantic_entry(sem_id)
    print(f"  Stored: {sem_id} (prompt={loaded_entry.prompt_text[:40]}...)")

    # Stats
    print(f"\n=== Stats ===\n  {store.stats()}")
    print("\n✅ Done")


if __name__ == "__main__":
    main()