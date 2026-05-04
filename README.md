# IntelliL3

**Intelligent three-tier object storage for LLM inference.**  
IntelliL3 manages KV-cache blocks, RAG chunks, semantic-cache entries, and agentic workflow artifacts across a three-tier hierarchy — GPU HBM (L1) → CPU RAM (L2) → S3-compatible object store (L3) — with policy-driven eviction, prefetching, and TTL-based retention.

---

## Why IntelliL3?

Large-scale LLM serving spends a significant fraction of request latency re-fetching KV-cache blocks and RAG documents that were recently computed but evicted from accelerator memory. IntelliL3 addresses this by:

- **Unifying** eight object types (KV-cache, RAG, semantic cache, agent workflows/steps/tools, plan cache, workflow traces) under one API
- **Routing** objects automatically between tiers based on access patterns and configured policies
- **Prefetching** the next likely blocks using session-history, semantic similarity, or workflow-topology signals
- **Retaining** agentic context across tool-call pauses via TTL-aware pinning

Benchmark results against a direct-S3 baseline (5 repetitions, mean ± std):

| Workload | Latency p95 baseline | Latency p95 IntelliL3 | Reduction |
|---|---:|---:|---:|
| Long Context 32 K | 324.9 ± 0.5 ms | 114.2 ± 0.1 ms | **−64.8 %** |
| RAG Heavy | 205.0 ± 0.5 ms | 65.5 ± 0.1 ms | **−68.1 %** |
| Multi-User Chat (50 users) | 181.5 ± 0.1 ms | 56.0 ± 0.1 ms | **−69.1 %** |
| Agentic ReAct Workflow | 208.5 ± 1.7 ms | 144.6 ± 2.2 ms | **−30.6 %** |
| Retail Workflow (TAU-like) | 151.8 ± 0.4 ms | 98.8 ± 0.4 ms | **−34.9 %** |
| Realistic RAG (50 queries) | 201.6 ± 1.4 ms | 93.3 ± 0.8 ms | **−53.7 %** |

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              LLM Inference Engine            │
│  (vLLM / SGLang / custom serving stack)     │
└──────────────────┬──────────────────────────┘
                   │  gRPC / Python API
┌──────────────────▼──────────────────────────┐
│           UnifiedObjectStore                │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Metadata │  │ Policies │  │Embeddings│  │
│  │  Store   │  │(eviction,│  │ Service  │  │
│  │(Bloom +  │  │ prefetch,│  │ (HNSW)   │  │
│  │ prefix   │  │   TTL)   │  │          │  │
│  │  tree +  │  └──────────┘  └──────────┘  │
│  │  HNSW)   │                               │
│  └──────────┘                               │
│                                             │
│  L1: GPU HBM  →  L2: CPU RAM  →  L3: S3   │
└─────────────────────────────────────────────┘
```

**Object types**: `KVCacheBlock` · `RAGObject` · `SemanticCacheEntry` · `AgentWorkflow` · `AgentStep` · `ToolCallArtifact` · `PlanCacheEntry` · `WorkflowTrace`

**Eviction policies**: LRU · Prefix-Reuse · Workflow-Aware

**Prefetch policies**: Session-History · Semantic-Similarity · Adaptive

---

## Installation

**Requirements**: Python ≥ 3.10, Docker (for local S3 via MinIO)

```bash
# Clone
git clone https://github.com/tmaxell/intellil3.git
cd intellil3

# Install (core)
pip install -e "."

# Install with embedding support (HNSW semantic search)
pip install -e ".[embeddings]"

# Install with gRPC server
pip install -e ".[grpc]"

# Install everything including dev tools
pip install -e ".[all]"
```

---

## Docker

The fastest way to run the full stack (IntelliL3 gRPC service + MinIO):

```bash
# Build image and start both services
docker compose -f deploy/docker/docker-compose.yaml up --build -d

# IntelliL3 gRPC → localhost:50051
# MinIO S3 API   → localhost:9000
# MinIO console  → http://localhost:9001  (minioadmin / minioadmin)
```

To point at real AWS S3 instead of MinIO, override the environment variables
without rebuilding the image:

```bash
L3_ENDPOINT_URL=https://s3.amazonaws.com \
L3_ACCESS_KEY=<key> \
L3_SECRET_KEY=<secret> \
L3_BUCKET=my-bucket \
docker compose -f deploy/docker/docker-compose.yaml up l3store -d
```

See [`docs/llm_integration_plan.md`](docs/llm_integration_plan.md) for a
step-by-step plan to wire IntelliL3 into vLLM, SGLang, and LangChain.

---

## Quick Start

**1. Start the full stack with Docker** *(recommended)*

```bash
docker compose -f deploy/docker/docker-compose.yaml up --build -d
```

**Or start MinIO only** and run l3store locally:

```bash
docker compose -f deploy/docker/docker-compose.yaml up minio -d
```

**2. Run the quickstart example**

```python
from l3store.core.object_store import UnifiedObjectStore
from l3store.core.types import KVCacheBlock
from l3store.storage.s3_backend import S3Backend
from l3store.utils.config import L3Config
import numpy as np

config = L3Config.from_yaml("configs/default.yaml")
backend = S3Backend(
    endpoint_url=config.storage.endpoint_url,
    access_key=config.storage.access_key,
    secret_key=config.storage.secret_key,
    bucket=config.storage.bucket,
)
store = UnifiedObjectStore(backend=backend, config=config)

# Store a KV-cache block
block = KVCacheBlock(model_name="llama3-8b", token_ids=list(range(16)), block_index=0)
key_states   = np.random.randn(32, 16, 8, 128).astype(np.float16)
value_states = np.random.randn(32, 16, 8, 128).astype(np.float16)

obj_id = store.put_kv_block(block, key_states, value_states)
loaded_block, loaded_keys, _ = store.get_kv_block(obj_id)
print(store.stats())
```

See [`examples/`](examples/) for more: eviction policies, embedding-based semantic search, and agentic artifact management.

---

## Running Tests

```bash
# All tests (635 tests, ~16 s)
PYTHONPATH=. pytest tests/ -q

# Specific test modules
PYTHONPATH=. pytest tests/test_concurrent_access.py -v
PYTHONPATH=. pytest tests/test_policy_combinations.py -v
PYTHONPATH=. pytest tests/test_large_scale.py -v
```

---

## Running Benchmarks

Benchmark results are written to `benchmarks/results/` (gitignored).

```bash
# Run all 10 experiments (5 repetitions each, ~3 min)
PYTHONPATH=. python -c "
from benchmarks.measurement.run_all_comparisons import run_all_comparisons
run_all_comparisons(
    config_glob='benchmarks/configs/exp*.yaml',
    output_dir='benchmarks/results',
    repetitions=5,
)
"

# Run a single experiment
PYTHONPATH=. python -m benchmarks.measurement.run_comparison \
  --config benchmarks/configs/exp09_retail_workflow_tau_like.yaml \
  --output-dir benchmarks/results/exp09 \
  --repetitions 5
```

Shell scripts for common experiment groups are in [`scripts/`](scripts/).

**Output per experiment** (in `benchmarks/results/<exp_name>/`):

| File | Contents |
|---|---|
| `raw_runs.json` | Per-request latencies for every repetition |
| `summary.json` | mean / std / min / max per metric per system |
| `comparison.json` | Baseline vs. candidate delta and effect size |
| `comparison.csv` | Machine-readable table for Excel / pandas |
| `comparison_report.md` | Human-readable report |

The top-level `benchmarks/results/comparison_index.csv` aggregates key metrics across all experiments.

---

## Project Structure

```
IntelliL3/
├── src/l3store/          # Core library
│   ├── core/             #   UnifiedObjectStore, types, ObjectMeta
│   ├── metadata/         #   MetadataStore (Bloom + prefix tree + HNSW)
│   ├── policies/         #   eviction/, prefetch/, retention/
│   ├── storage/          #   S3Backend, MemoryBackend
│   ├── semantic_cache/   #   Embedding-based deduplication
│   ├── agent/            #   Agentic workflow tracking
│   ├── embeddings/       #   EmbeddingService (HNSW index)
│   ├── api/              #   gRPC service definitions
│   └── utils/            #   Config, logging
├── benchmarks/
│   ├── configs/          #   YAML experiment definitions (exp01–exp10)
│   ├── workloads/        #   Synthetic workload generators
│   ├── baselines/        #   System implementations (LRU, L3-full, agentic)
│   ├── measurement/      #   Runner, aggregation, comparison reports
│   └── data/             #   Benchmark datasets (RAG corpus, retail tasks)
├── tests/                # 635 pytest tests
├── configs/              # default.yaml (storage + object config)
├── deploy/docker/        # docker-compose.yaml (MinIO)
├── examples/             # Quickstart scripts
└── scripts/              # Shell wrappers for benchmark runs
```

---

## Configuration

Edit `configs/default.yaml` to point at your S3-compatible store:

```yaml
storage:
  backend: s3
  endpoint_url: "http://localhost:9000"   # MinIO or AWS S3
  access_key: "minioadmin"
  secret_key: "minioadmin"
  bucket: "l3-llm-store"
  region: "us-east-1"
```

All object-type key prefixes, block sizes, and policy parameters are configurable in the same file.

---

## License

MIT
