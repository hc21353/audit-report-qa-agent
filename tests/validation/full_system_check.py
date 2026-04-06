#!/usr/bin/env python3
"""
전체 시스템 검증 스크립트
- RAG 구조
- 데이터베이스
- 앱 실행
- 파일 시스템
- 테스팅
- CI/Docker 설정
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime

# tests/validation/ -> parent.parent.parent = 프로젝트 루트
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# 색상 및 기호
# ─────────────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
CHECKMARK = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"
WARNING = f"{YELLOW}⚠{RESET}"
INFO = f"{BLUE}ℹ{RESET}"

# ─────────────────────────────────────────────────────────────────────────────
# 1. RAG 구조 검증
# ─────────────────────────────────────────────────────────────────────────────

def check_rag_structure():
    print(f"\n{BLUE}{'='*70}")
    print("1. RAG 구조 검증")
    print(f"{'='*70}{RESET}\n")
    
    tests = []
    
    # Config 로드
    try:
        from src.config import load_config
        config = load_config()
        tests.append((True, "Config 로드"))
    except Exception as e:
        tests.append((False, f"Config 로드: {e}"))
    
    # Agents 초기화
    try:
        from src.agents.state import SearchResult, SubQuery, initial_state
        tests.append((True, "Agent state (SearchResult, SubQuery, initial_state)"))
    except Exception as e:
        tests.append((False, f"Agent state: {e}"))
    
    # Tools 검증
    try:
        from src.tools.hybrid_search import hybrid_search
        from src.tools.db_fetch import get_full_content
        tests.append((True, "Tools import 가능 (hybrid_search, get_full_content)"))
    except Exception as e:
        tests.append((False, f"Tools: {e}"))
    
    # LLM 백엔드 검증
    try:
        from src.agents.llm import create_llm, resolve_backend, get_system_prompt
        tests.append((True, "LLM 팩토리 가능 (create_llm, resolve_backend, get_system_prompt)"))
    except Exception as e:
        tests.append((False, f"LLM: {e}"))
    
    for success, msg in tests:
        symbol = CHECKMARK if success else CROSS
        print(f"  {symbol} {msg}")
    
    return all(t[0] for t in tests)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 데이터베이스 검증
# ─────────────────────────────────────────────────────────────────────────────

def check_database():
    print(f"\n{BLUE}{'='*70}")
    print("2. 데이터베이스 검증")
    print(f"{'='*70}{RESET}\n")
    
    results = []
    
    # SQLite DB
    db_path = ROOT / "db" / "audit_reports.db"
    if db_path.exists():
        print(f"  {CHECKMARK} DB 파일 존재: {db_path.stat().st_size / (1024*1024):.2f} MB")
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # 테이블 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"  {CHECKMARK} 테이블: {len(tables)}개 - {', '.join(tables[:3])}...")
            
            # chunks 행 수
            cursor.execute("SELECT COUNT(*) FROM chunks")
            count = cursor.fetchone()[0]
            print(f"  {CHECKMARK} Chunks: {count:,}개")
            
            # 연도별 분포
            cursor.execute("""
                SELECT fiscal_year, COUNT(*) as cnt 
                FROM chunks 
                GROUP BY fiscal_year 
                ORDER BY fiscal_year
            """)
            years = cursor.fetchall()
            print(f"  {CHECKMARK} 연도별 분포: {len(years)}개 연도")
            for year, cnt in years[:3]:
                print(f"      {year}: {cnt:,}개")
            
            conn.close()
            results.append(True)
        except Exception as e:
            print(f"  {CROSS} DB 쿼리 실패: {e}")
            results.append(False)
    else:
        print(f"  {CROSS} DB 파일 없음: {db_path}")
        results.append(False)
    
    # ChromaDB 벡터스토어
    vector_dir = ROOT / "db" / "vectorstore" / "chroma"
    if vector_dir.exists():
        print(f"  {CHECKMARK} ChromaDB 디렉토리 존재")
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.PersistentClient(
                path=str(vector_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            collections = client.list_collections()
            print(f"  {CHECKMARK} Collections: {len(collections)}개")
            for col in collections:
                print(f"      {col.name}: {col.count()}개 벡터")
            results.append(True)
        except Exception as e:
            print(f"  {CROSS} ChromaDB 접근 실패: {e}")
            results.append(False)
    else:
        print(f"  {CROSS} ChromaDB 디렉토리 없음: {vector_dir}")
        results.append(False)
    
    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 앱 실행 검증
# ─────────────────────────────────────────────────────────────────────────────

def check_app():
    print(f"\n{BLUE}{'='*70}")
    print("3. Streamlit 앱 검증")
    print(f"{'='*70}{RESET}\n")
    
    results = []
    
    # Streamlit 파일
    app_file = ROOT / "src" / "app.py"
    if app_file.exists():
        print(f"  {CHECKMARK} app.py 존재")
        
        # Streamlit import 가능성
        try:
            import streamlit
            print(f"  {CHECKMARK} Streamlit 설치됨 ({streamlit.__version__})")
            results.append(True)
        except ImportError:
            print(f"  {CROSS} Streamlit 미설치")
            results.append(False)
        
        # 파일 크기
        size_kb = app_file.stat().st_size / 1024
        print(f"  {CHECKMARK} app.py 크기: {size_kb:.1f} KB")
    else:
        print(f"  {CROSS} app.py 없음")
        results.append(False)
    
    # 설정 파일
    config_dir = ROOT / "config"
    config_files = ["app.yaml", "agents.yaml", "runtime.yaml", "tools.yaml"]
    for cfg_file in config_files:
        if (config_dir / cfg_file).exists():
            print(f"  {CHECKMARK} {cfg_file}")
        else:
            print(f"  {CROSS} {cfg_file} 없음")
            results.append(False)
    
    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 파일 시스템 검증
# ─────────────────────────────────────────────────────────────────────────────

def check_file_system():
    print(f"\n{BLUE}{'='*70}")
    print("4. 파일 시스템 검증")
    print(f"{'='*70}{RESET}\n")
    
    results = []
    
    # 필수 디렉토리
    dirs = {
        "src": "소스 코드",
        "config": "설정 파일",
        "db": "데이터베이스",
        "parsed_data": "파싱된 데이터",
        "tests": "테스트",
        "docs": "문서",
    }
    
    for dirname, desc in dirs.items():
        path = ROOT / dirname
        if path.exists() and path.is_dir():
            count = len(list(path.glob("**/*")))
            print(f"  {CHECKMARK} {dirname:15} ({desc:12}): {count} items")
            results.append(True)
        else:
            print(f"  {CROSS} {dirname:15} ({desc:12}): 없음")
            results.append(False)
    
    # 빌드 스크립트
    print(f"\n  {INFO} 빌드 스크립트:")
    build_scripts = ["src/build_db.py", "src/build_vectordb.py", "src/semantic_chunking.py"]
    for script in build_scripts:
        if (ROOT / script).exists():
            print(f"    {CHECKMARK} {script}")
        else:
            print(f"    {CROSS} {script} 없음")
            results.append(False)
    
    # data 폴더 (삭제되어야 함)
    if (ROOT / "data").exists():
        print(f"  {WARNING} data/ 폴더 아직 존재 (삭제 필요)")
        results.append(False)
    else:
        print(f"  {CHECKMARK} data/ 폴더 제거됨 (정상)")
    
    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 테스트 검증
# ─────────────────────────────────────────────────────────────────────────────

def check_testing():
    print(f"\n{BLUE}{'='*70}")
    print("5. 테스트 검증")
    print(f"{'='*70}{RESET}\n")
    
    results = []
    
    # pytest 테스트 파일
    test_dir = ROOT / "tests" / "pytest"
    if test_dir.exists():
        test_files = list(test_dir.glob("test_*.py"))
        print(f"  {CHECKMARK} pytest 테스트: {len(test_files)}개 파일")
        for tf in test_files:
            print(f"      - {tf.name}")
        results.append(len(test_files) > 0)
    else:
        print(f"  {CROSS} pytest 디렉토리 없음")
        results.append(False)
    
    # 검증 스크립트
    validation_dir = ROOT / "tests" / "validation"
    if validation_dir.exists():
        script_files = list(validation_dir.glob("*.py"))
        print(f"  {CHECKMARK} 검증 스크립트: {len(script_files)}개 파일")
        results.append(len(script_files) > 0)
    else:
        print(f"  {CROSS} validation 디렉토리 없음")
        results.append(False)
    
    # pytest 설치
    try:
        import pytest
        print(f"  {CHECKMARK} pytest 설치됨 ({pytest.__version__})")
        results.append(True)
    except ImportError:
        print(f"  {CROSS} pytest 미설치")
        results.append(False)
    
    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CI/CD 검증
# ─────────────────────────────────────────────────────────────────────────────

def check_cicd():
    print(f"\n{BLUE}{'='*70}")
    print("6. CI/CD 설정 검증")
    print(f"{'='*70}{RESET}\n")
    
    results = []
    
    # GitHub Actions 워크플로우
    ci_file = ROOT / ".github" / "workflows" / "ci.yml"
    if ci_file.exists():
        print(f"  {CHECKMARK} CI 워크플로우 존재")
        with open(ci_file) as f:
            content = f.read()
            if "python-version: '3.11'" in content or 'python-version: "3.11"' in content:
                print(f"      Python 3.11 설정됨")
            if "pytest" in content:
                print(f"      pytest 스텝 포함")
        results.append(True)
    else:
        print(f"  {CROSS} CI 워크플로우 없음")
        results.append(False)
    
    # Docker 설정
    docker_file = ROOT / "Dockerfile"
    if docker_file.exists():
        print(f"  {CHECKMARK} Dockerfile 존재")
        with open(docker_file) as f:
            content = f.read()
            if "3.11" in content:
                print(f"      Python 3.11 기반 이미지")
            if "streamlit" in content:
                print(f"      Streamlit 설치됨")
        results.append(True)
    else:
        print(f"  {CROSS} Dockerfile 없음")
        results.append(False)
    
    # docker-compose
    compose_file = ROOT / "docker-compose.yml"
    if compose_file.exists():
        print(f"  {CHECKMARK} docker-compose.yml 존재")
        with open(compose_file) as f:
            content = f.read()
            if "ollama" in content.lower():
                print(f"      Ollama 서비스 정의됨")
        results.append(True)
    else:
        print(f"  {CROSS} docker-compose.yml 없음")
        results.append(False)
    
    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# 7. 의존성 검증
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies():
    print(f"\n{BLUE}{'='*70}")
    print("7. 주요 의존성 검증")
    print(f"{'='*70}{RESET}\n")
    
    packages = {
        "chromadb": "벡터스토어",
        "langgraph": "멀티에이전트",
        "langchain": "LLM 프레임워크",
        "streamlit": "UI",
        "pandas": "데이터 처리",
        "lxml": "HTML 파싱",
    }
    
    results = []
    for package, desc in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "?")
            print(f"  {CHECKMARK} {package:15} ({desc:12}): {version}")
            results.append(True)
        except ImportError:
            print(f"  {CROSS} {package:15} ({desc:12}): 미설치")
            results.append(False)
    
    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# 8. 경로 및 import 검증
# ─────────────────────────────────────────────────────────────────────────────

def check_paths_and_imports():
    print(f"\n{BLUE}{'='*70}")
    print("8. 경로 및 Import 검증")
    print(f"{'='*70}{RESET}\n")
    
    results = []
    
    # 경로 확인
    paths = {
        "db/audit_reports.db": "SQLite DB",
        "db/vectorstore/chroma": "ChromaDB",
        "parsed_data": "파싱 데이터",
        "config": "설정",
    }
    
    for relpath, desc in paths.items():
        full_path = ROOT / relpath
        if full_path.exists():
            print(f"  {CHECKMARK} {relpath:30} ✓")
            results.append(True)
        else:
            print(f"  {CROSS} {relpath:30} ✗ 없음")
            results.append(False)
    
    # 주요 Import 테스트
    print(f"\n  {INFO} 주요 모듈 Import:")
    imports = [
        ("src.config", "Config 로더"),
        ("src.db", "DB 관리"),
        ("src.tools.hybrid_search", "하이브리드 검색"),
        ("src.agents.graph", "Agent Graph"),
    ]
    
    for module, desc in imports:
        try:
            __import__(module)
            print(f"    {CHECKMARK} {module:30} ({desc})")
            results.append(True)
        except Exception as e:
            print(f"    {CROSS} {module:30} 실패: {str(e)[:30]}")
            results.append(False)
    
    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BLUE}")
    print("=" * 70)
    print("🔍 Samsung Audit Report QA Agent - 전체 시스템 검증")
    print("=" * 70)
    print(f"{RESET}")
    print(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"루트: {ROOT}")
    
    checks = [
        ("RAG 구조", check_rag_structure),
        ("데이터베이스", check_database),
        ("Streamlit 앱", check_app),
        ("파일 시스템", check_file_system),
        ("테스트", check_testing),
        ("CI/CD", check_cicd),
        ("의존성", check_dependencies),
        ("경로 & Import", check_paths_and_imports),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n{RED}❌ {name} 검증 중 오류: {e}{RESET}")
            results[name] = False
    
    # 최종 요약
    print(f"\n{BLUE}{'='*70}")
    print("최종 검증 결과")
    print(f"{'='*70}{RESET}\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        symbol = CHECKMARK if result else CROSS
        print(f"  {symbol} {name:20}: {'통과' if result else '실패'}")
    
    print(f"\n  {BLUE}총 결과: {passed}/{total} 통과{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}✅ 모든 시스템이 정상 작동합니다!{RESET}\n")
        return 0
    else:
        print(f"\n{RED}❌ 일부 항목이 실패했습니다. 위 결과를 확인하세요.{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
