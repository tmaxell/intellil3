# IntelliL3 — LLM Inference Integration Plan

This document describes how to connect IntelliL3 to a real LLM inference engine
so that KV-cache blocks, RAG chunks, and semantic-cache entries are managed by
IntelliL3 instead of being discarded when GPU memory is full.

---

## Current state

IntelliL3 is a **standalone gRPC service** (`port 50051`) that exposes a fully
implemented object-store API:

```
PutKVBlock / GetKVBlock / DeleteKVBlock / ListKVBlocks
PutRAGObject / GetRAGObject
PutSemanticEntry / GetSemanticEntry / SearchSimilarPrompts
PrefetchBatch / GetStats
```

The Python client (`l3store.api.grpc_client.L3Client`) wraps every RPC in a
typed, synchronous call. An async variant is the first item in the work plan
below.

---

## Integration architecture

```
┌─────────────────────────────────────────────────────────┐
│                 LLM Inference Engine                     │
│   (vLLM / SGLang / TensorRT-LLM / custom)               │
│                                                         │
│  ┌──────────────────┐     ┌────────────────────────┐   │
│  │  KV-Cache Engine │────▶│   IntelliL3 plugin /   │   │
│  │  (BlockAllocator,│     │   monkey-patch shim     │   │
│  │   CacheEngine)   │◀────│   (L3Client wrapper)   │   │
│  └──────────────────┘     └──────────┬─────────────┘   │
│                                       │ gRPC            │
└───────────────────────────────────────┼─────────────────┘
                                        │
                         ┌──────────────▼──────────────────┐
                         │      IntelliL3 gRPC service      │
                         │  (docker compose up  /  k8s pod) │
                         └──────────────┬──────────────────┘
                                        │ S3 API
                         ┌──────────────▼──────────────────┐
                         │   MinIO / AWS S3 / GCS / Azure   │
                         └─────────────────────────────────┘
```

---

## Work items

### Phase 1 — Async Python client (1–2 days)

**File**: `src/l3store/api/async_client.py`

The current `L3Client` is synchronous and would block the event loop of async
inference frameworks (vLLM, SGLang, Triton). Replace it with an `asyncio`-native
counterpart using `grpcio-asyncio`:

```python
import grpc.aio
from l3store.api.proto import l3_service_pb2_grpc as svc

class AsyncL3Client:
    async def put_kv_block(self, block, key_states, value_states) -> str: ...
    async def get_kv_block(self, object_id) -> ...: ...
    async def prefetch_batch(self, decisions) -> ...: ...
```

**Why needed first**: every downstream integration depends on a non-blocking
client to avoid adding latency inside the inference hot-path.

---

### Phase 2 — vLLM integration (3–5 days)

**Target class**: `vllm.worker.cache_engine.CacheEngine`

vLLM manages KV blocks through `swap_in` / `swap_out` / `copy` operations
between GPU and CPU tensors. IntelliL3 adds a third tier (L3 ↔ CPU):

**File**: `integrations/vllm/l3_cache_engine.py`

```python
from vllm.worker.cache_engine import CacheEngine
from l3store.api.async_client import AsyncL3Client
from l3store.core.types import KVCacheBlock

class L3CacheEngine(CacheEngine):
    """Drop-in replacement that offloads evicted CPU blocks to IntelliL3."""

    def __init__(self, *args, l3_endpoint: str = "localhost:50051", **kwargs):
        super().__init__(*args, **kwargs)
        self._l3 = AsyncL3Client(l3_endpoint)

    async def swap_out(self, src_to_dst: dict[int, int]) -> None:
        """Standard GPU→CPU swap, then async offload CPU blocks to L3."""
        await super().swap_out(src_to_dst)
        for cpu_block_id in src_to_dst.values():
            block, k, v = self._read_cpu_block(cpu_block_id)
            await self._l3.put_kv_block(block, k, v)

    async def swap_in(self, src_to_dst: dict[int, int]) -> None:
        """Try L3 first; fall back to standard CPU→GPU copy on miss."""
        for l3_id, gpu_slot in src_to_dst.items():
            result = await self._l3.get_kv_block(l3_id)
            if result:
                block, k, v = result
                self._write_gpu_block(gpu_slot, k, v)
            else:
                await super().swap_in({l3_id: gpu_slot})
```

**Wiring**: pass `worker_cls="integrations.vllm.l3_cache_engine.L3CacheEngine"`
to `vllm.LLM(...)` or set it in the engine config.

**Key challenge**: vLLM's internal block IDs are integers, not content-addressed
hashes. The shim must maintain a `{cpu_block_id → l3_object_id}` map to
correlate evictions with L3 lookups.

---

### Phase 3 — SGLang integration (2–3 days)

**Target class**: `sglang.srt.mem_cache.radix_cache.RadixCache`

SGLang uses a radix tree for prefix-aware KV reuse. IntelliL3 natively
supports prefix-tree metadata (`MetadataStore`), so the integration can be
deeper than vLLM:

**File**: `integrations/sglang/l3_radix_cache.py`

```python
from sglang.srt.mem_cache.radix_cache import RadixCache
from l3store.api.async_client import AsyncL3Client

class L3RadixCache(RadixCache):
    """Extends SGLang's radix cache with L3 offload for evicted nodes."""

    def evict(self, num_tokens: int) -> int:
        evicted = super().evict(num_tokens)
        for node in self._last_evicted_nodes:
            asyncio.ensure_future(
                self._l3.put_kv_block(node.kv_block, node.key_states, node.val_states)
            )
        return evicted

    async def match_prefix(self, token_ids: list[int]):
        result = await super().match_prefix(token_ids)
        if result.num_matched < len(token_ids):
            # Query IntelliL3 prefix tree for a longer match
            l3_hit = await self._l3.search_prefix(token_ids)
            if l3_hit:
                await self._promote_from_l3(l3_hit)
        return result
```

**Benefit**: SGLang's prefix-reuse semantic aligns exactly with IntelliL3's
`PrefixReuseEviction` policy — eviction decisions can be driven by L3 metadata.

---

### Phase 4 — RAG pipeline integration (1–2 days)

Most RAG stacks (LangChain, LlamaIndex, Haystack) have a pluggable vector-store
interface. IntelliL3 can act as one:

**File**: `integrations/langchain/l3_vector_store.py`

```python
from langchain.vectorstores.base import VectorStore

class IntelliL3VectorStore(VectorStore):
    def add_texts(self, texts, embeddings, **kwargs):
        for text, emb in zip(texts, embeddings):
            obj = RAGObject(chunk_text=text, ...)
            self._l3.put_rag_object(obj, emb)

    def similarity_search_by_vector(self, embedding, k=4):
        matches = self._l3.search_similar_prompts(embedding, top_k=k)
        return [self._l3.get_rag_object(oid) for oid, _ in matches]
```

The `SearchSimilarPrompts` RPC is already implemented in IntelliL3 and uses
an HNSW index internally.

---

### Phase 5 — Kubernetes / production deployment (2–3 days)

**Files**: `deploy/k8s/`

```
deploy/k8s/
├── namespace.yaml
├── minio-statefulset.yaml      # or use AWS S3 (no minio needed)
├── l3store-deployment.yaml     # 2-3 replicas, ReadWriteMany PVC for metadata
├── l3store-service.yaml        # ClusterIP on port 50051
├── l3store-hpa.yaml            # autoscale on CPU/memory
└── configmap.yaml              # L3 config, injected as volume
```

Key considerations:
- **Metadata store** (`MetadataStore`) is in-process and not replicated.  
  For multi-replica setups, either use a single leader pod or externalize  
  the Bloom filter + prefix tree to Redis.
- **gRPC load balancing**: use `grpc.io/load_balancing_policy: round_robin`  
  in the k8s Service headless DNS mode.
- **TLS**: add `grpc.ssl_server_credentials` in `grpc_server.serve()` and  
  mount a TLS secret.

---

### Phase 6 — Observability (1 day)

- Expose Prometheus metrics via `prometheus_client` on `:9090/metrics`  
  (cache_hit_rate, eviction_count, l3_read_latency_seconds histogram)
- Add OpenTelemetry tracing spans around `PutKVBlock` / `GetKVBlock`
- Wire gRPC health protocol (`grpc_health_checking`) so k8s liveness probes
  use the standard gRPC health check instead of a Python subprocess

---

## Effort summary

| Phase | What | Effort |
|-------|------|--------|
| 1 | Async Python client | 1–2 days |
| 2 | vLLM `CacheEngine` shim | 3–5 days |
| 3 | SGLang `RadixCache` shim | 2–3 days |
| 4 | LangChain / LlamaIndex vector store | 1–2 days |
| 5 | Kubernetes manifests | 2–3 days |
| 6 | Prometheus + OpenTelemetry | 1 day |
| **Total** | | **10–16 days** |

Phases 1 → 2 → 3 must be done in order.  
Phases 4, 5, 6 are independent and can be parallelised.

---

## Quick integration test (without code changes to the inference engine)

While the shims above are being built, you can validate the round-trip
immediately using the existing synchronous client:

```python
# On the vLLM / SGLang host, after `pip install l3-llm-store[grpc]`
from l3store.api.grpc_client import L3Client
from l3store.core.types import KVCacheBlock
import numpy as np

with L3Client("l3store:50051") as client:   # container hostname
    block = KVCacheBlock(model_name="llama3-8b", token_ids=list(range(16)), block_index=0)
    k = np.random.randn(32, 16, 8, 128).astype(np.float16)
    v = np.random.randn(32, 16, 8, 128).astype(np.float16)

    obj_id = client.put_kv_block(block, k, v)
    result = client.get_kv_block(obj_id)
    assert result is not None
    print("Round-trip OK:", obj_id)
    print(client.get_stats())
```
