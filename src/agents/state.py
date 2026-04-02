"""
state.py - LangGraph 상태 정의

모든 에이전트 노드가 공유하는 상태(State) 스키마.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class SearchResult:
    """검색된 청크 하나"""
    content: str = ""
    score: float = 0.0
    year: int = 0
    section_path: str = ""
    content_type: str = "text"
    chunk_id: int = 0
    csv_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "score": self.score,
            "year": self.year,
            "section_path": self.section_path,
            "content_type": self.content_type,
            "chunk_id": self.chunk_id,
            "csv_refs": self.csv_refs,
        }


@dataclass
class SubQuery:
    """재작성된 서브 쿼리"""
    query: str = ""
    years: list[int] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    strategy: str = ""  # original | keyword | accounting_term


# LangGraph에서는 TypedDict를 사용하지만, 호환성을 위해
# 여기서는 dict 기반으로 정의하고 graph.py에서 TypedDict로 래핑

def initial_state(user_query: str) -> dict:
    """초기 상태 생성"""
    return {
        # 사용자 입력
        "user_query": user_query,

        # orchestrator 출력
        "intent": "",                  # simple_lookup | comparison | trend | calculation | general
        "extracted_years": [],         # 질문에서 추출된 연도
        "extracted_sections": [],      # 질문에서 추출된 섹션
        "sub_questions": [],           # 분해된 서브 질문

        # query_rewriter 출력
        "rewritten_queries": [],       # SubQuery 리스트

        # retriever 출력
        "search_results": [],          # SearchResult 리스트
        "csv_data": {},                # csv_path → 테이블 내용 매핑

        # analyst 출력
        "answer": "",                  # 최종 답변
        "sources": [],                 # 인용 소스
        "calculations": [],            # 계산 결과
        "charts": [],                  # 차트 데이터
        "needs_more_search": False,    # 추가 검색 필요 여부
        "additional_query": "",        # 추가 검색 쿼리

        # 메타
        "iteration": 0,
        "max_iterations": 5,
        "errors": [],
    }
