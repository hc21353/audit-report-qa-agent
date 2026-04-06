# ============================================================
# Dockerfile - Samsung Audit Report RAG 시스템
# ============================================================

# 파이썬 3.11 슬림 버전 사용 (PyTorch, ONNX 호환성 최적화)
FROM python:3.11-slim

# 시스템 의존성 설치 (ML 라이브러리 컴파일 시 필요)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 먼저 복사 (캐시 최적화)
COPY requirements.txt .

# pip 업그레이드 및 패키지 설치
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# Streamlit 포트 개방
EXPOSE 8501

# 기본 실행 명령어 (docker-compose에서 덮어쓸 수 있음)
# Streamlit 서버 구성 (헬스체크용)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]