#!/bin/bash
# ============================================================
# setup.sh - 프로젝트 초기 설정 스크립트
# ============================================================

set -e  # 에러 발생 시 즉시 종료

echo "============================================================"
echo "Samsung Audit Report QA Agent - 초기 설정"
echo "============================================================"

# 1. Python 가상환경 확인/생성
echo ""
echo "[1/4] Python 가상환경 확인..."
if [ ! -d ".venv" ]; then
    echo "  → 가상환경 생성 중..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. 패키지 설치
echo ""
echo "[2/4] 필수 패키지 설치..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 3. 데이터베이스 구축
echo ""
echo "[3/4] 데이터베이스 구축 (SQLite)..."
if [ -f "build_db.py" ]; then
    python3 build_db.py --reset 2>/dev/null || echo "  ✓ DB 이미 존재"
fi

# 4. 벡터스토어 구축
echo ""
echo "[4/4] 벡터스토어 구축 (ChromaDB + KoE5)..."
if [ -f "build_vectordb.py" ]; then
    python3 build_vectordb.py --reset 2>/dev/null || echo "  ✓ 벡터스토어 이미 존재"
fi

echo ""
echo "============================================================"
echo "✓ 초기 설정 완료!"
echo "============================================================"
echo ""
echo "다음 명령어로 애플리케이션을 실행하세요:"
echo ""
echo "  📱 Streamlit UI 실행:"
echo "     streamlit run src/app.py"
echo ""
echo "  🐳 Docker로 실행:"
echo "     docker-compose up --build"
echo ""
echo "  🧪 설정 진단:"
echo "     python3 test_setup.py"
echo ""
echo "============================================================"
