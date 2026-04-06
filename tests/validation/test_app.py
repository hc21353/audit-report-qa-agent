#!/usr/bin/env python3
"""Streamlit 앱 초기화 테스트"""

import sys
from pathlib import Path

# tests/validation/ -> parent.parent.parent = 프로젝트 루트
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

print('=' * 60)
print('Streamlit App 초기화 테스트')
print('=' * 60)
print()

# 1. 설정 로드 테스트
print("[1/4] 설정 로드...")
try:
    from src.config import load_config
    config = load_config()
    print('  ✓ 설정 로드 성공')
    print(f'    - DB 경로: {config.db_path}')
    project_name = config.app.get("project", {}).get("name", "unknown")
    print(f'    - 프로젝트: {project_name}')
except Exception as e:
    print(f'  ✗ 설정 로드 실패: {e}')
    sys.exit(1)

# 2. 데이터베이스 초기화 테스트
print("\n[2/4] 데이터베이스 연결...")
try:
    from src.db import AuditDB
    db = AuditDB(config.db_path)
    stats = db.conn.execute('SELECT COUNT(*) as cnt FROM chunks').fetchone()
    print(f'  ✓ 데이터베이스 연결 성공')
    print(f'    - 총 청크: {stats["cnt"]:,}개')
except Exception as e:
    print(f'  ✗ DB 연결 실패: {e}')
    sys.exit(1)

# 3. 벡터스토어 초기화 테스트
print("\n[3/4] 벡터스토어 초기화...")
try:
    import chromadb
    from chromadb.config import Settings
    
    vector_dir = 'db/vectorstore/chroma'
    client = chromadb.PersistentClient(
        path=vector_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    collections = client.list_collections()
    print(f'  ✓ 벡터스토어 초기화 성공')
    print(f'    - 컬렉션: {len(collections)}개')
    for col in collections:
        print(f'      - {col.name}: {col.count():,} 문서')
except Exception as e:
    print(f'  ⚠️ 벡터스토어 오류 (비필수): {e}')

# 4. LLM 연결 테스트
print("\n[4/4] LLM 서버 연결...")
try:
    import requests
    response = requests.get('http://localhost:11434/api/tags', timeout=2)
    if response.status_code == 200:
        models = response.json().get('models', [])
        print(f'  ✓ Ollama 서버 연결 성공')
        print(f'    - 설치된 모델: {len(models)}개')
        for model in models[:3]:
            print(f'      - {model.get("name")}')
    else:
        print(f'  ⚠️ Ollama 상태 코드: {response.status_code}')
except Exception as e:
    print(f'  ✗ Ollama 연결 실패: {e}')
    print('    → Ollama 서버를 실행해야 합니다!')

print()
print('=' * 60)
print('✓ 모든 핵심 모듈이 정상 로드되었습니다!')
print('=' * 60)
print()
print('다음 명령어로 Streamlit을 실행하세요:')
print('  streamlit run src/app.py')
print()
