"""
tools/ - 에이전트 Tool 모듈

검색:
  - hybrid_search:           벡터 + BM25 하이브리드 (RRF 결합)
  - structured_query:        h1~h6 스키마 기반 구조 검색
  - list_available_sections: 섹션 목록 조회
  - csv_reader_tool:         [TABLE_CSV] CSV 동적 로드
  - web_search:              DuckDuckGo 웹 검색 (최신 정보 보완)
  - get_full_content:        청크 전체 내용 조회 (잘린 내용 전체 가져오기)

분석:
  - calculator:         수식 계산 + 재무비율 프리셋
  - chart_generator:    Plotly 차트 생성
"""

from src.tools.hybrid_search import hybrid_search
from src.tools.structured_query import structured_query, list_available_sections
from src.tools.csv_reader_tool import csv_reader_tool
from src.tools.calculator import calculator
from src.tools.chart_generator import chart_generator
from src.tools.web_search import web_search
from src.tools.db_fetch import get_full_content

RETRIEVER_TOOLS = [hybrid_search, structured_query, list_available_sections, csv_reader_tool, web_search, get_full_content]
ANALYST_TOOLS = [calculator, chart_generator, csv_reader_tool]
ALL_TOOLS = RETRIEVER_TOOLS + ANALYST_TOOLS
