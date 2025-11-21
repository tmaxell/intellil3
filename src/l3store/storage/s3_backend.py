from __future__ import annotations

import logging
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from l3store.storage.backend import StorageBackend

logger = logging.getLogger(__name__)


class S3Backend(StorageBackend):

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str = "us-east-1",
    ):
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=BotoConfig(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"},
                max_pool_connections=50,
            ),
        )
        logger.info("S3Backend: endpoint=%s bucket=%s", endpoint_url, bucket)

    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self._bucket)
            logger.info("Created bucket %s", self._bucket)

    def put(self, key: str, data: bytes, metadata: Optional[dict] = None) -> None:
        extra = {}
        if metadata:
            extra["Metadata"] = {k: str(v) for k, v in metadata.items()}
        self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentLength=len(data), **extra
        )

    def get(self, key: str) -> Optional[bytes]:
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            return resp["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def delete(self, key: str) -> bool:
        if not self.exists(key):
            return False
        self._client.delete_object(Bucket=self._bucket, Key=key)
        return True

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False

    def list_keys(self, prefix: str = "") -> list[str]:
        keys = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def get_metadata(self, key: str) -> Optional[dict]:
        try:
            resp = self._client.head_object(Bucket=self._bucket, Key=key)
            return {
                "content_length": resp["ContentLength"],
                "last_modified": resp["LastModified"].isoformat(),
                "user_metadata": resp.get("Metadata", {}),
            }
        except ClientError:
            return None