from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    backend: str = "s3"
    endpoint_url: str = "http://localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "l3-llm-store"
    region: str = "us-east-1"


class KVCacheConfig(BaseModel):
    block_size: int = 16
    key_prefix: str = "kv/"


class RAGConfig(BaseModel):
    key_prefix: str = "rag/"


class SemanticCacheConfig(BaseModel):
    key_prefix: str = "sem/"


class AgentObjectsConfig(BaseModel):
    workflows_prefix: str = "agent/workflows/"
    steps_prefix: str = "agent/steps/"
    tools_prefix: str = "agent/tools/"
    plans_prefix: str = "agent/plans/"
    traces_prefix: str = "agent/traces/"


class ObjectsConfig(BaseModel):
    kv_cache: KVCacheConfig = Field(default_factory=KVCacheConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    semantic_cache: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)
    agent: AgentObjectsConfig = Field(default_factory=AgentObjectsConfig)


class L3Config(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    objects: ObjectsConfig = Field(default_factory=ObjectsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> L3Config:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    @classmethod
    def default(cls) -> L3Config:
        return cls()
