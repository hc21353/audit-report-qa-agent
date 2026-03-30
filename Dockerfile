# 파이썬 3.10 슬림 버전 사용 (빌드 속도와 용량 최적화)
FROM python:3.10-slim

# 시스템 의존성 설치 (lxml 등 컴파일 시 필요)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 패키지 설치 (캐시 활용을 위해 requirements 먼저 복사)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# Streamlit 기본 포트 개방
EXPOSE 8501

# 실행 명령어 (나중에 docker-compose에서 덮어쓰기 가능)
CMD ["streamlit", "run", "src/ui/app.py"]