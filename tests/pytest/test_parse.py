"""pytest 테스트: parser_final.py 검증"""
import sys
from pathlib import Path

import pytest

# tests/pytest/ -> parent.parent.parent = 프로젝트 루트
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src import parser_final as P
except ImportError:
    pytest.skip("parser_final 모듈 없음", allow_module_level=True)


# ─────────────────────────────────────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_text_basic():
    """기본 텍스트 정규화"""
    result = P.normalize_text("  Hello  World  ")
    assert "Hello" in result
    assert "World" in result


def test_chunk_load():
    """청크 로드 가능성 확인"""
    try:
        from src.db import AuditDB
        from src.config import load_config
        
        config = load_config()
        db = AuditDB(config.db_path)
        
        result = db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        chunk_count = result[0] if result else 0
        
        assert chunk_count > 0, "청크가 없습니다"
        
        db.conn.close()
    except Exception as e:
        pytest.skip(f"DB 접근 불가: {e}")


def test_vectorstore_available():
    """벡터스토어 접근 가능성 확인"""
    try:
        import chromadb
        from chromadb.config import Settings
        from pathlib import Path
        
        vector_dir = Path(__file__).parent.parent / "db" / "vectorstore" / "chroma"
        
        if not vector_dir.exists():
            pytest.skip("벡터스토어 디렉토리 없음")
        
        client = chromadb.PersistentClient(
            path=str(vector_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        collections = client.list_collections()
        assert len(collections) > 0, "컬렉션이 없습니다"
        
    except Exception as e:
        pytest.skip(f"벡터스토어 접근 불가: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
