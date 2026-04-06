#!/usr/bin/env python3
"""
경로 검증 스크립트 - 모든 import와 파일 참조 확인

실행: python test_paths.py
"""

import sys
import os
from pathlib import Path

# 루트 디렉토리 (tests/validation/ -> parent.parent.parent = 프로젝트 루트)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

print("=" * 70)
print("경로 검증 - 모든 Import & 파일 참조 확인")
print("=" * 70)

# 1. 루트 경로 확인
print("\n[1] 프로젝트 루트 경로")
print(f"  ROOT: {ROOT}")
print(f"  ✓ 존재: {ROOT.exists()}")

# 2. 주요 디렉토리 확인
print("\n[2] 필수 디렉토리")
required_dirs = [
    "src",
    "config",
    "db",
    "tests",
    "docs",
]
for d in required_dirs:
    path = ROOT / d
    exists = "✓" if path.exists() else "✗"
    print(f"  {exists} {d}: {path}")

# 3. 설정 파일 확인
print("\n[3] 설정 파일 (config/)")
config_files = ["app.yaml", "agents.yaml", "runtime.yaml", "tools.yaml"]
for f in config_files:
    path = ROOT / "config" / f
    exists = "✓" if path.exists() else "✗"
    print(f"  {exists} {f}: {path}")

# 4. 데이터베이스 파일 확인
print("\n[4] 데이터 파일 (db/)")
data_files = {
    "audit_reports.db": "SQLite DB",
    "vectorstore/chroma": "ChromaDB 벡터스토어",
}
for f, desc in data_files.items():
    path = ROOT / "db" / f
    exists = "✓" if path.exists() else "✗"
    size_info = ""
    if path.exists() and path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        size_info = f" ({size_mb:.2f} MB)"
    print(f"  {exists} {f}: {desc}{size_info}")

# 5. Python import 검증
print("\n[5] Python Import 검증")
test_imports = [
    ("src.config", "Config loader"),
    ("src.db", "Database manager"),
    ("src.agents.graph", "LangGraph"),
    ("src.agents.orchestrator", "Orchestrator node"),
    ("src.agents.retriever", "Retriever node"),
    ("src.agents.analyst", "Analyst node"),
]

for module, desc in test_imports:
    try:
        __import__(module)
        print(f"  ✓ {module} ({desc})")
    except ImportError as e:
        print(f"  ✗ {module} ({desc}): {e}")
    except Exception as e:
        print(f"  ⚠️ {module} ({desc}): {type(e).__name__}: {e}")

# 6. Config 로드 검증
print("\n[6] Config 로드 검증")
try:
    from src.config import load_config
    config = load_config()
    print(f"  ✓ Config 로드 성공")
    print(f"    - Project: {config.app.get('project', {}).get('name', 'N/A')}")
    print(f"    - DB path: {config.db_path}")
    print(f"    - Years: {len(config.years)} 년도 ({config.years[0]}-{config.years[-1]})")
    print(f"    - Chunking strategy: {config.active_chunking}")
except Exception as e:
    print(f"  ✗ Config 로드 실패: {e}")

# 7. 데이터베이스 연결 검증
print("\n[7] 데이터베이스 연결 검증")
try:
    from src.db import AuditDB
    from src.config import load_config
    config = load_config()
    db = AuditDB(config.db_path)
    count = db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"  ✓ DB 연결 성공")
    print(f"    - 총 청크: {count:,}개")
    db.conn.close()
except Exception as e:
    print(f"  ✗ DB 연결 실패: {e}")

# 8. 벡터스토어 검증
print("\n[8] 벡터스토어 검증")
try:
    import chromadb
    from chromadb.config import Settings
    
    vector_dir = ROOT / "db" / "vectorstore" / "chroma"
    client = chromadb.PersistentClient(
        path=str(vector_dir),
        settings=Settings(anonymized_telemetry=False)
    )
    collections = client.list_collections()
    print(f"  ✓ ChromaDB 연결 성공")
    print(f"    - 컬렉션 수: {len(collections)}")
    for col in collections:
        print(f"      - {col.name}: {col.count():,} 벡터")
except Exception as e:
    print(f"  ⚠️ ChromaDB 오류: {e}")

# 9. build_db.py 경로 확인
print("\n[9] 빌드 스크립트 경로")
build_scripts = [
    "build_db.py",
    "build_vectordb.py",
    "src/build_db.py",
    "src/build_vectordb.py",
]
for script in build_scripts:
    path = ROOT / script
    if path.exists():
        print(f"  ✓ {script}")
    else:
        print(f"  - {script} (없음)")

print("\n" + "=" * 70)
print("✅ 경로 검증 완료")
print("=" * 70)
