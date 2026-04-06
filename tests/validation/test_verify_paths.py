#!/usr/bin/env python3
"""폴더 이동 후 모든 경로 & 의존성 검증"""

import sys
from pathlib import Path

print("=" * 70)
print("폴더 이동 후 경로 & 의존성 검증")
print("=" * 70)

# 1. tests/validation/check_chunks.py 경로
print("\n[1] tests/validation/check_chunks.py 경로")
# tests/validation/ -> parent.parent.parent = 프로젝트 루트
validation_check_chunks = Path(__file__).resolve()
print(f"  파일 위치: {validation_check_chunks}")
print(f"  parent: {validation_check_chunks.parent}")
print(f"  parent.parent.parent: {validation_check_chunks.parent.parent.parent}")

root = validation_check_chunks.parent.parent.parent
file_path = root / "parsed_data" / "chunks" / "semantic_chunks_tagged.jsonl"
print(f"  참조 파일: {file_path}")
print(f"  ✓ 존재: {file_path.exists()}" if file_path.exists() else f"  ✗ 없음: {file_path}")

# 2. tests/validation/check_db.py 경로
print("\n[2] tests/validation/check_db.py 경로")
validation_check_db = Path(__file__).parent / "check_db.py"
print(f"  파일 위치: {validation_check_db}")

# tests/validation/ -> parent.parent.parent = 프로젝트 루트
root = validation_check_db.parent.parent.parent
db_path = root / "db" / "audit_reports.db"
vector_dir = root / "db" / "vectorstore" / "chroma"
log_path = root / "db" / "validation_log.jsonl"

print(f"  ROOT: {root}")
print(f"  DB_PATH: {db_path}")
print(f"  ✓ 존재: {db_path.exists()}" if db_path.exists() else f"  ✗ 없음: {db_path}")
print(f"  VECTOR_DIR: {vector_dir}")
print(f"  ✓ 존재: {vector_dir.exists()}" if vector_dir.exists() else f"  ✗ 없음: {vector_dir}")

# 3. tests/pytest/test_parse.py 경로
print("\n[3] tests/pytest/test_parse.py 경로")
pytest_test_parse = Path(__file__).parent.parent / "pytest" / "test_parse.py"
print(f"  파일 위치: {pytest_test_parse}")

sys_path_insert = pytest_test_parse.parent.parent.parent
print(f"  sys.path 추가: {sys_path_insert}")
print(f"  ✓ 루트 맞음: {sys_path_insert == Path.cwd()}")

# import 경로
print(f"\n  Import 검증:")
sys.path.insert(0, str(sys_path_insert))

try:
    from src.db import AuditDB
    print(f"    ✓ from src.db import AuditDB")
except ImportError as e:
    print(f"    ✗ from src.db import AuditDB: {e}")

try:
    from src.config import load_config
    print(f"    ✓ from src.config import load_config")
except ImportError as e:
    print(f"    ✗ from src.config import load_config: {e}")

try:
    from src.agents.graph import build_graph
    print(f"    ✓ from src.agents.graph import build_graph")
except ImportError as e:
    print(f"    ✗ from src.agents.graph import build_graph: {e}")

# 4. CI에서 실행 시나리오
print("\n[4] CI 실행 경로 (from project root)")
print(f"  cd /Users/hyeju/audit-report-qa-agent")
print(f"  pytest tests/test_parse.py -v")
print(f"    → tests/test_parse.py 경로: {Path('tests/test_parse.py').exists()}")
print(f"  python scripts/check_chunks.py")
print(f"    → scripts/check_chunks.py 경로: {Path('scripts/check_chunks.py').exists()}")
print(f"  python scripts/check_db.py")
print(f"    → scripts/check_db.py 경로: {Path('scripts/check_db.py').exists()}")

# 5. 중요 검증
print("\n[5] 중요 경로 검증")
checks = [
    ("parsed_data/chunks/semantic_chunks_tagged.jsonl", Path("parsed_data/chunks/semantic_chunks_tagged.jsonl")),
    ("db/audit_reports.db", Path("db/audit_reports.db")),
    ("db/vectorstore/chroma", Path("db/vectorstore/chroma")),
    ("config/app.yaml", Path("config/app.yaml")),
    ("src/db.py", Path("src/db.py")),
    ("src/config.py", Path("src/config.py")),
]

all_ok = True
for desc, path in checks:
    exists = path.exists()
    symbol = "✓" if exists else "✗"
    print(f"  {symbol} {desc}")
    if not exists:
        all_ok = False

print("\n" + "=" * 70)
if all_ok:
    print("✅ 모든 경로가 정상입니다!")
else:
    print("⚠️ 일부 경로에 문제가 있습니다!")
print("=" * 70)
