import boto3
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