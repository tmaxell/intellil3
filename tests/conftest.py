import boto3
import numpy as np
import pytest
from moto import mock_aws

from l3store.core.object_store import UnifiedObjectStore
from l3store.metadata.metadata_store import MetadataStore
from l3store.storage.s3_backend import S3Backend
from l3store.utils.config import L3Config, StorageConfig


@pytest.fixture
def test_config() -> L3Config:
    return L3Config(
        storage=StorageConfig(
            endpoint_url="http://localhost:5000",
            access_key="testing",
            secret_key="testing",
            bucket="test-l3-store",
            region="us-east-1",
        )
    )


@pytest.fixture
def mock_s3(test_config):
    with mock_aws():
        conn = boto3.client("s3", region_name="us-east-1")
        conn.create_bucket(Bucket=test_config.storage.bucket)

        backend = S3Backend(
            endpoint_url=test_config.storage.endpoint_url,
            access_key=test_config.storage.access_key,
            secret_key=test_config.storage.secret_key,
            bucket=test_config.storage.bucket,
            region=test_config.storage.region,
        )
        backend._client = conn
        yield backend


@pytest.fixture
def store(mock_s3, test_config) -> UnifiedObjectStore:
    metadata = MetadataStore(expected_items=1000, fp_rate=0.01)
    return UnifiedObjectStore(backend=mock_s3, config=test_config, metadata=metadata)


# ---------------------------------------------------------------------------
# Parametrized store fixtures for different scale scenarios
# ---------------------------------------------------------------------------

@pytest.fixture(params=[
    pytest.param({"expected_items": 100, "fp_rate": 0.01}, id="small"),
    pytest.param({"expected_items": 10_000, "fp_rate": 0.01}, id="medium"),
    pytest.param({"expected_items": 100_000, "fp_rate": 0.001}, id="large"),
])
def store_at_scale(request, mock_s3, test_config) -> UnifiedObjectStore:
    """Store fixture parametrized across three Bloom-filter scale configurations."""
    params = request.param
    metadata = MetadataStore(
        expected_items=params["expected_items"],
        fp_rate=params["fp_rate"],
    )
    return UnifiedObjectStore(backend=mock_s3, config=test_config, metadata=metadata)


# ---------------------------------------------------------------------------
# Shared helpers used across multiple test modules
# ---------------------------------------------------------------------------

def make_embedding(dim: int = 3, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def make_kv_tensors(
    num_heads: int = 4,
    seq_len: int = 16,
    head_dim: int = 8,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (num_heads, seq_len, head_dim)
    return (
        rng.standard_normal(shape).astype(np.float32),
        rng.standard_normal(shape).astype(np.float32),
    )