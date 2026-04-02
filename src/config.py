"""
config.py - YAML 설정 파일 로더

사용법:
    from src.config import load_config
    cfg = load_config()
    print(cfg.app["project"]["name"])
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_yaml(filename: str) -> dict:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Config:
    app: dict = field(default_factory=dict)
    agents: dict = field(default_factory=dict)
    tools: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)

    # 자주 쓰는 값 바로 접근
    @property
    def db_path(self) -> str:
        return self.runtime.get("database", {}).get("path", "./data/audit_reports.db")

    @property
    def years(self) -> list[int]:
        return self.app.get("project", {}).get("years", [])

    @property
    def active_chunking(self) -> str:
        return self.app.get("chunking", {}).get("active_strategy", "section_based")

    @property
    def chunking_config(self) -> dict:
        strategy = self.active_chunking
        return self.app.get("chunking", {}).get("strategies", {}).get(strategy, {})

    @property
    def active_embedding(self) -> str:
        return self.app.get("embedding", {}).get("active_model", "multilingual-e5-large")

    @property
    def embedding_config(self) -> dict:
        model = self.active_embedding
        return self.app.get("embedding", {}).get("models", {}).get(model, {})

    @property
    def parsing_dir(self) -> str:
        return self.app.get("parsing", {}).get("output_dir", "./data/parsed_md")


def load_config() -> Config:
    return Config(
        app=_load_yaml("app.yaml"),
        agents=_load_yaml("agents.yaml"),
        tools=_load_yaml("tools.yaml"),
        runtime=_load_yaml("runtime.yaml"),
    )


if __name__ == "__main__":
    cfg = load_config()
    print(f"Project: {cfg.app['project']['name']}")
    print(f"DB path: {cfg.db_path}")
    print(f"Years: {cfg.years}")
    print(f"Active chunking: {cfg.active_chunking}")
    print(f"Active embedding: {cfg.active_embedding}")
