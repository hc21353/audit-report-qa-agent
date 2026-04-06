#!/usr/bin/env python3
"""프로젝트 설정 진단 스크립트"""

import sys
import sqlite3
from pathlib import Path

def test_database():
    """SQLite DB 테스트"""
    print("\n=== 1. SQLite 데이터베이스 ===")
    db_path = Path("db/audit_reports.db")
    
    if not db_path.exists():
        print(f"✗ DB 파일 없음: {db_path}")
        return False
    
    print(f"✓ DB 파일 존재: {db_path}")
    print(f"  크기: {db_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # 테이블 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"  테이블: {tables}")
        
        # chunks 테이블 행 수
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        print(f"  chunks 행 수: {chunk_count:,}")
        
        conn.close()
        return True
    except Exception as e:
        print(f"✗ DB 접근 실패: {e}")
        return False

def test_vectorstore():
    """ChromaDB 벡터스토어 테스트"""
    print("\n=== 2. ChromaDB 벡터스토어 ===")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        vector_dir = Path("db/vectorstore/chroma")
        print(f"벡터 스토어 경로: {vector_dir}")
        print(f"디렉토리 존재: {vector_dir.exists()}")
        
        if not vector_dir.exists():
            print("⚠️ 벡터스토어 디렉토리가 없습니다!")
            return False
        
        # ChromaDB 연결
        client = chromadb.PersistentClient(
            path=str(vector_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        print("✓ ChromaDB 클라이언트 연결 성공")
        
        # 컬렉션 확인
        collections = client.list_collections()
        print(f"✓ 컬렉션 개수: {len(collections)}")
        
        for col in collections:
            print(f"  - '{col.name}': {col.count()} 문서")
        
        if len(collections) == 0:
            print("⚠️ 컬렉션이 없습니다! build_vectordb.py를 실행해야 합니다.")
            return False
        
        return True
    except Exception as e:
        print(f"✗ ChromaDB 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_model():
    """KoE5 임베딩 모델 테스트"""
    print("\n=== 3. KoE5 임베딩 모델 ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("로딩 중... (처음 실행 시 모델 다운로드 ~1GB)")
        model = SentenceTransformer("nlpai-lab/KoE5")
        dim = model.get_sentence_embedding_dimension()
        
        print(f"✓ KoE5 모델 로드 성공")
        print(f"  - 임베딩 차원: {dim}d")
        
        # 간단한 테스트
        embedding = model.encode("test", normalize_embeddings=True)
        print(f"  - 테스트 임베딩 생성 성공 (크기: {len(embedding)})")
        
        return True
    except Exception as e:
        print(f"✗ KoE5 모델 오류: {e}")
        return False

def test_config_files():
    """설정 파일 테스트"""
    print("\n=== 4. 설정 파일 ===")
    
    config_files = ["config/app.yaml", "config/agents.yaml", "config/runtime.yaml", "config/tools.yaml"]
    all_exist = True
    
    for cf in config_files:
        exists = Path(cf).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {cf}")
        if not exists:
            all_exist = False
    
    # YAML 파싱 테스트
    try:
        import yaml
        with open("config/app.yaml") as f:
            config = yaml.safe_load(f)
        print("✓ YAML 파일 파싱 성공")
        return all_exist
    except Exception as e:
        print(f"✗ YAML 파싱 오류: {e}")
        return False

def test_requirements():
    """Requirements 확인"""
    print("\n=== 5. Python 패키지 ===")
    
    # (설치 모듈명, 표시 이름, 설명)
    required_packages = [
        ("yaml", "pyyaml", "YAML 설정"),
        ("langgraph", "langgraph", "멀티에이전트 워크플로우"),
        ("langchain", "langchain", "LLM 프레임워크"),
        ("streamlit", "streamlit", "UI"),
        ("chromadb", "chromadb", "벡터스토어"),
        ("sentence_transformers", "sentence-transformers", "임베딩"),
        ("torch", "torch", "딥러닝"),
    ]
    
    all_installed = True
    for import_name, display_name, desc in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {display_name} ({desc})")
        except ImportError:
            print(f"✗ {display_name} ({desc}) - 설치 필요!")
            all_installed = False
    
    return all_installed

def test_ollama():
    """Ollama 서버 연결 테스트"""
    print("\n=== 6. Ollama LLM 서버 ===")
    
    try:
        import requests
        
        # Ollama 기본 포트
        url = "http://localhost:11434/api/tags"
        response = requests.get(url, timeout=2)
        
        if response.status_code == 200:
            print("✓ Ollama 서버 연결 성공 (http://localhost:11434)")
            models = response.json()
            if "models" in models:
                print(f"  설치된 모델 ({len(models['models'])}개):")
                for model in models["models"][:5]:
                    print(f"    - {model.get('name', 'unknown')}")
            return True
        else:
            print(f"✗ Ollama 응답 오류: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Ollama 서버 연결 불가 (http://localhost:11434)")
        print("  → Ollama를 실행해주세요: ollama serve 또는 'Ollama' 앱 실행")
        return False
    except Exception as e:
        print(f"✗ Ollama 테스트 오류: {e}")
        return False

def main():
    print("=" * 60)
    print("삼성 감사보고서 RAG 시스템 설정 진단")
    print("=" * 60)
    
    results = {
        "DB": test_database(),
        "VectorStore": test_vectorstore(),
        "EmbeddingModel": test_embedding_model(),
        "ConfigFiles": test_config_files(),
        "Requirements": test_requirements(),
        "Ollama": test_ollama(),
    }
    
    print("\n" + "=" * 60)
    print("진단 결과 요약")
    print("=" * 60)
    
    for component, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {component}: {'정상' if status else '문제'}")
    
    all_ok = all(results.values())
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ 모든 시스템이 정상 구성되어 있습니다!")
        print("\n다음 명령어로 실행하세요:")
        print("  streamlit run src/app.py")
    else:
        print("⚠️ 몇 가지 문제가 있습니다:")
        if not results["DB"]:
            print("  → build_db.py를 실행하여 DB를 생성해야 합니다")
        if not results["VectorStore"]:
            print("  → build_vectordb.py를 실행하여 벡터스토어를 생성해야 합니다")
        if not results["Ollama"]:
            print("  → Ollama 서버를 실행해야 합니다")
        if not results["Requirements"]:
            print("  → pip install -r requirements.txt 를 실행해야 합니다")
    print("=" * 60)

if __name__ == "__main__":
    main()
