"""

개선 사항 요약
─────────────────────────────────────────────────────────
1. [숫자 저장 전략]
   - table CSV: 첫 번째 컬럼(계정명)은 str, 나머지는 float (parse_numeric 적용)
   - 금액 표시용 문자열이 필요할 때는 이미 저장된 float에 단위(unit)를 붙여서 표현
   - 계산이 필요한 경우 float 컬럼을 그대로 사용 → 별도 변환 불필요
   - 단위(unit)는 table_manifest.csv 에 문자열로 저장, 수치 환산은 저장 시점이 아닌
     분석 시점에 tool calling 등으로 처리 (단위별 scale_factor 참고용으로만 주석 제공)

2. [혼용 컬럼 처리]
   - % / 주(株) / 금액이 섞인 테이블:
     · 각 셀을 parse_numeric() 시도 → float 변환 가능하면 float, 아니면 str 보존
     · 컬럼 헤더에 '%' 또는 '주' 키워드가 있으면 col_type 을 manifest 에 기록
   - 1행×1열 테이블(텍스트 박스 역할) → CSV 저장 안 하고 MD에 일반 문장으로 삽입

3. [span 태그 분절 대응]
   - <span>2</span><span>.5 현금및현금성자산</span> 처럼 숫자·점이
     별도 span으로 쪼개진 경우를 재조립 (merge_inline_spans 함수)
   - 조립 후 → 기존 split_embedded_headers / merge_split_numbering_segments 파이프라인에 투입
   - 이로써 "2.5 현금및현금성자산" 같은 섹션 제목이 올바르게 헤더 후보로 인식됨

4. [테이블 필터링]
   - class="TABLE" 인 <table> 만 처리
   - class="nb" 는 명시적으로 제외: nb 클래스는 감사보고서 HTM에서 주로 레이아웃용
     보조 테이블(여백·서명란 등)로 쓰이며 실제 재무 데이터를 담지 않음
   - numeric_like_ratio 제거 (manifest 컬럼에서도 삭제)
   - source_table_class 컬럼 제거

5. [table_manifest.csv 컬럼]
   table_index | csv_path | row_count | col_count | unit | col_types
   · col_types: 각 컬럼이 '금액' / '%' / '주' / '혼용' / '텍스트' 인지 자동 분류한 JSON

6. [불필요 CSV 제거]
   - segments.csv, candidate_list.csv 저장 안 함
   - final_hierarchy.csv 저장 안 함 (tree.txt 로 충분)
   - 중간 결과 CSV 일절 저장 안 함

7. [thead/tbody 완전 파싱]
   - pd.read_html 실패 시 manual_parse_table() 이 thead + tbody 를 모두 순회
   - td / th 모두 포함, colspan/rowspan 은 반복 채움으로 처리

8. [Markdown 출력 - 문서 흐름 순서 유지]
   - DOM 순서대로 h태그→텍스트→테이블 참조를 섞어 출력
   - h1~h6 → Markdown # 헤더로 변환 (h1=#, h2=##, ..., h6=######)
   - class="SECTION-1" 등 SECTION 클래스가 있으면 depth 기반 추가 헤딩 처리
   - 1×1 텍스트 박스 → 일반 문장으로 인라인 삽입
   - class="TABLE" 테이블 → [TABLE: tables/table_NNNN.csv  단위: XXX] 참조 (흐름 중 삽입)
   - class="nb" 테이블 → 무시 (렌더링하지 않음)

9. [쉼표(,) 구분 오파싱 수정]
   - 주석 참조 "4, 6, 31" 같은 쉼표 구분 목록이 금액 파싱에서 "4631"로 합산되는 문제 수정
   - parse_numeric(): 쉼표가 3자리 간격이 아니면 천단위 구분자가 아닌 것으로 판단 → None 반환
   - is_amount_comma(): 쉼표가 천단위 패턴인지 검증하는 헬퍼 추가

10. [줄 분절 보존]
    - <p> 안에 여러 의미 단위(회사명, 수신인, 섹션 제목 등)가 br 또는 span 경계로
      나뉜 경우 각각 별도 줄로 출력 (merge_inline_spans 에서 br → \n 마커 보존)
─────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import re
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from collections import Counter, defaultdict

import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag


# =========================================================
# 1. 데이터 구조
# =========================================================

@dataclass
class Segment:
    """HTML에서 추출한 최소 텍스트 단위 (한 블록 = 한 p/h 태그)"""
    segment_id: int
    source_tag: str        # 원본 태그명 (p, h2, span 등)
    block_index: int
    text: str


@dataclass
class HeaderCandidate:
    """섹션 헤더 후보 정보"""
    segment_id: int
    block_index: int
    text: str
    normalized_text: str
    pattern_type: str
    raw_prefix: str
    title_score: float
    context_before: str = ""
    context_after: str = ""
    candidate_type: str = "header_candidate"   # header_candidate / uncertain

    candidate_group: str = "unknown"           # major / middle / minor / subminor / unknown
    section_score: float = 0.0
    is_real_section: bool = False

    inferred_depth: Optional[int] = None
    parent_segment_id: Optional[int] = None
    status: str = "unresolved"
    reason: str = ""


@dataclass
class InferredNode:
    """계층 추론 결과 노드"""
    segment_id: int
    block_index: int
    text: str
    pattern_type: str
    depth: Optional[int]
    parent_segment_id: Optional[int]
    status: str
    reason: str


@dataclass
class DocumentPatternProfile:
    """문서 전체의 헤더 패턴 통계 프로파일"""
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    transition_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    dominant_transitions: Dict[str, str] = field(default_factory=dict)
    pattern_order_hint: Dict[str, int] = field(default_factory=dict)


@dataclass
class TableExportInfo:
    """
    테이블 경로 관리용 manifest 행.
    · numeric_like_ratio, source_table_class 제거
    · col_types 추가: 각 컬럼의 데이터 유형 JSON 문자열
    """
    table_index: int
    csv_path: str
    row_count: int
    col_count: int
    unit: Optional[str]          # 단위 문자열 (예: "백만원", "천원", "%")
    col_types: str               # JSON 문자열 {"계정과목": "텍스트", "당기": "금액", ...}


# =========================================================
# 2. 패턴 정의
# =========================================================

PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("section_class_1", re.compile(r"^\s*주석\s*$")),
    ("note_numeric",    re.compile(r"^\s*주석\s*(\d+)[\.\)]?\s*")),
    ("decimal_3",       re.compile(r"^\s*(\d+\.\d+\.\d+)[\.\)]?\s+")),
    ("decimal_2",       re.compile(r"^\s*(\d+\.\d+)[\.\)]?\s+")),
    ("decimal_1",       re.compile(r"^\s*(\d+)[\.\)]\s+")),
    ("paren_num",       re.compile(r"^\s*(\((\d+)\))\s+")),
    ("korean_alpha",    re.compile(r"^\s*([가-하])\.\s+")),
    ("circled_num",     re.compile(r"^\s*([①②③④⑤⑥⑦⑧⑨⑩])\s+")),
    ("upper_alpha",     re.compile(r"^\s*([A-Z])\.\s+")),
    ("roman",           re.compile(r"^\s*([ivxlcdm]+)[\.\)]\s+", re.IGNORECASE)),
]

PATTERN_BASE_DEPTH_HINT = {
    "section_class_1": 1,
    "note_numeric":    2,
    "decimal_1":       2,
    "decimal_2":       3,
    "decimal_3":       4,
    "paren_num":       4,
    "korean_alpha":    4,
    "circled_num":     5,
    "upper_alpha":     5,
    "roman":           2,
}

PATTERN_GROUP_CANDIDATES: Dict[str, List[str]] = {
    "section_class_1": ["major"],
    "note_numeric":    ["major", "middle"],
    "decimal_1":       ["major", "middle"],
    "decimal_2":       ["middle", "minor"],
    "decimal_3":       ["minor", "subminor"],
    "roman":           ["middle"],
    "korean_alpha":    ["minor", "subminor"],
    "paren_num":       ["subminor"],
    "circled_num":     ["subminor"],
    "upper_alpha":     ["subminor"],
}

GROUP_TO_DEPTH = {
    "major":    1,
    "middle":   2,
    "minor":    3,
    "subminor": 4,
    "unknown":  99,
}

TITLE_LIKE_ENDINGS = (
    "사항", "정책", "회계정책", "회계처리방침", "회계추정", "가정",
    "수익인식", "금융상품", "금융자산", "금융부채", "리스", "재고자산",
    "유형자산", "무형자산", "현금및현금성자산", "매출채권", "매입채무",
    "차입금", "충당부채", "정부보조금", "주당이익", "승인", "법인세",
    "외화환산", "손상", "배당금", "자본금", "종업원급여",
    "매각예정분류자산집단", "순확정급여부채", "순확정급여부채(자산)",
    "재무제표 작성기준",
)

NON_HEADER_PATTERNS = [
    re.compile(r"^\s*\(?주\)?\s*\d+\s*$"),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*[\(\[]?\d+[\)\]]?\s*$"),
    re.compile(r"^\s*제\s*\d+\s*기\s*$"),
    re.compile(r"^\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*$"),
]

NOISE_PATTERNS = [
    re.compile(r"^\s*서울특별시"),
    re.compile(r"^\s*삼\s*[일정회법인]+\s*회\s*계\s*법\s*인"),
    re.compile(r"^\s*삼정회계법인"),
    re.compile(r"^\s*대\s*표\s*이\s*사"),
    re.compile(r"^\s*이 감사보고서는"),
    re.compile(r"^\s*주주 및 이사회 귀중"),
    re.compile(r"^\s*삼성전자주식회사\s*$"),
]


# =========================================================
# 3. 전처리 유틸
# =========================================================

def normalize_text(text: str) -> str:
    """다중 공백·특수 공백 정규화 (줄바꿈 보존)"""
    text = text.replace("\xa0", " ")
    text = text.replace("\u3000", " ")
    text = text.replace("ㆍ", "ㆍ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def normalize_inline_text(text: str) -> str:
    """줄바꿈 포함 모든 공백을 단일 스페이스로 통합 (인라인용)"""
    text = text.replace("\xa0", " ")
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_noise_text(text: str) -> bool:
    """의미 없는 노이즈 텍스트 여부 판단"""
    t = normalize_inline_text(text)
    if not t:
        return True
    for p in NON_HEADER_PATTERNS:
        if p.match(t):
            return True
    for p in NOISE_PATTERNS:
        if p.match(t):
            return True
    return False


def is_probable_non_header(text: str) -> bool:
    """헤더가 아닐 가능성이 높은 텍스트인지 판단 (긴 문장, 종결어미 등)"""
    t = normalize_inline_text(text)
    if not t:
        return True
    if is_noise_text(t):
        return True
    if len(t) > 130:
        return True
    if re.search(r"(입니다|합니다|하였다|됩니다|있습니다|한다|합니다\.)$", t):
        return True
    return False


def is_continuation_header(text: str) -> bool:
    """
    '계속' 연속 표시가 붙은 헤더 감지.
    ex) "2.1 금융상품, 계속:"  → True
    ex) "2. 중요한 회계처리방침, 계속 :"  → True   ← 공백+콜론 형태
    ex) "주석 3 (계속)"        → True
    """
    t = normalize_inline_text(text)
    if "계속" not in t:
        return False

    # ", 계속 :" / ", 계속:" / "계속 :" / "(계속)" 등 말미 패턴
    if re.search(r"[,，]?\s*계속\s*:?\s*$", t):
        return True
    if re.search(r"[\(（]\s*계속\s*[\)）]", t):
        return True

    continuation_starts = [
        r"^\d+\.\d+\.\d+",
        r"^\d+\.\d+",
        r"^\d+\.",
        r"^주석\s*\d+",
        r"^[가-하]\.",
        r"^\(\d+\)",
    ]
    if any(re.match(p, t) for p in continuation_starts):
        return True
    return False


def clean_header_text(text: str) -> str:
    """헤더 텍스트에서 '계속' 꼬리 제거 및 잡공백 정리.

    처리 대상 패턴 예시:
      "2. 중요한 회계처리방침, 계속 :"  → "2. 중요한 회계처리방침"
      "2.1 금융상품, 계속:"             → "2.1 금융상품"
      "주석 3 (계속)"                   → "주석 3"
    """
    t = normalize_inline_text(text)
    # ,/， + 선택적 공백 + 계속 + 선택적 공백 + 선택적 콜론(:) + 후행 공백
    t = re.sub(r"[,，]?\s*계속\s*:?\s*$", "", t).strip()
    # "(계속)" 단독 괄호 형태도 제거
    t = re.sub(r"\s*[\(（]\s*계속\s*[\)）]\s*$", "", t).strip()
    # 잔여 후행 콤마 정리 ("금융상품," → "금융상품")
    t = re.sub(r"[,，]\s*$", "", t).strip()
    t = re.sub(r"\s+:", ":", t)
    return t


def classify_pattern(text: str) -> Tuple[str, str]:
    """텍스트에서 헤더 패턴 분류 (decimal_2, korean_alpha 등)"""
    for pattern_type, pattern in PATTERNS:
        m = pattern.match(text)
        if m:
            raw_prefix = m.group(1) if m.lastindex else pattern_type
            return pattern_type, raw_prefix
    return "none", ""


# =========================================================
# 4. span 분절 재조립 (핵심 신규 기능)
# =========================================================

def merge_inline_spans(tag: Tag) -> str:
    """
    PDF→HTML 변환 시 span으로 쪼개진 텍스트를 재조립한다.

    예시:
      <span>2</span><span>.5 현금및현금성자산</span>
      → "2.5 현금및현금성자산"

    전략:
      1) 모든 자식 노드의 텍스트를 순서대로 이어붙임
      2) 숫자·점·괄호로 시작하는 fragment 는 앞 조각과 공백 없이 결합
      3) 그 외 fragment 는 공백으로 결합
    """
    fragments: List[str] = []

    for child in tag.descendants:
        if isinstance(child, NavigableString):
            frag = str(child)
            frag = frag.replace("\xa0", " ").replace("\u3000", " ")
            # 줄바꿈을 공백으로 (br 처리와 별개)
            frag = re.sub(r"[\r\n]+", " ", frag)
            if frag.strip():
                fragments.append(frag)
        elif isinstance(child, Tag) and child.name == "br":
            # <br> 은 줄바꿈 마커로 보존
            fragments.append("\n")

    if not fragments:
        return ""

    # 조각 결합: 숫자 뒤에 ".숫자" 가 이어지면 붙임
    result_chars: List[str] = []
    for i, frag in enumerate(fragments):
        if i == 0:
            result_chars.append(frag)
            continue

        prev = result_chars[-1] if result_chars else ""
        prev_stripped = prev.rstrip()

        # 앞 조각이 숫자로 끝나고 현재 조각이 '.'으로 시작 → 공백 없이 결합
        if re.search(r"\d$", prev_stripped) and re.match(r"^\.\d", frag.lstrip()):
            result_chars.append(frag.lstrip())
        # 현재 조각이 숫자·점으로만 구성된 접두사(분절된 번호) → 앞 공백 제거
        elif re.match(r"^\s*[\d\.]+\s", frag) and re.search(r"\d$", prev_stripped):
            result_chars.append(frag.lstrip())
        else:
            result_chars.append(frag)

    merged = "".join(result_chars)
    # 다중 공백 통합 (줄바꿈 제외)
    merged = re.sub(r"[ \t]+", " ", merged)
    return merged.strip()


# =========================================================
# 5. 점수 계산
# =========================================================

def compute_title_score(
    text: str,
    pattern_type: str,
    prev_text: str = "",
    next_text: str = "",
) -> float:
    """헤더 후보의 제목 점수 계산 (0.0 ~ 1.0)"""
    score = 0.0
    norm = normalize_inline_text(text)

    if pattern_type != "none":
        score += 0.45
    if 2 <= len(norm) <= 70:
        score += 0.15
    if len(norm) <= 40:
        score += 0.10
    if any(keyword in norm for keyword in TITLE_LIKE_ENDINGS):
        score += 0.10
    if not re.search(r"(입니다|합니다|하였다|됩니다|있습니다|한다)$", norm):
        score += 0.05
    if ":" in norm or norm.endswith(":"):
        score += 0.03

    if prev_text:
        prev_norm = normalize_inline_text(prev_text)
        if prev_norm == "주석":
            score += 0.07

    if next_text:
        next_norm = normalize_inline_text(next_text)
        if len(next_norm) > len(norm) + 8:
            score += 0.05

    if re.match(r"^[가-하]\.\s+", norm) and len(norm) > 50:
        score -= 0.20
    if re.match(r"^\(\d+\)\s+", norm) and len(norm) > 60:
        score -= 0.20

    return max(0.0, min(1.0, score))


def candidate_groups_for_pattern(pattern_type: str) -> List[str]:
    return PATTERN_GROUP_CANDIDATES.get(pattern_type, ["unknown"])


def choose_candidate_group(pattern_type: str, prev_pattern_type: str = "none") -> str:
    """직전 패턴 전이를 고려해 후보 그룹(major/middle/minor/subminor) 결정"""
    groups = candidate_groups_for_pattern(pattern_type)
    if len(groups) == 1:
        return groups[0]

    if pattern_type == "note_numeric":
        return "major" if prev_pattern_type in {"section_class_1", "none"} else "middle"
    if pattern_type == "decimal_1":
        return "middle" if prev_pattern_type in {"note_numeric", "section_class_1"} else "major"
    if pattern_type == "decimal_2":
        return "minor" if prev_pattern_type in {"decimal_1", "note_numeric"} else "middle"
    if pattern_type == "decimal_3":
        return "subminor" if prev_pattern_type in {"decimal_2", "korean_alpha", "paren_num"} else "minor"
    if pattern_type == "korean_alpha":
        return "subminor" if prev_pattern_type in {"decimal_2", "decimal_3"} else "minor"
    return groups[0]


def compute_section_authenticity_score(
    text: str,
    pattern_type: str,
    title_score: float,
    prev_text: str = "",
    next_text: str = "",
    prev_pattern_type: str = "none",
) -> Tuple[float, str, bool]:
    """
    '진짜 섹션인지 아닌지' 판별 점수.
    반환: (score, reason_string, is_real)
    """
    score = title_score
    reasons: List[str] = []
    norm = normalize_inline_text(text)

    if pattern_type != "none":
        score += 0.15
        reasons.append("has_pattern")
    if pattern_type in PATTERN_BASE_DEPTH_HINT:
        score += 0.05
        reasons.append("depth_hint_known")
    if norm.endswith(("사항", "정책", "기준", "구성", "평가", "추정", "위험", "공시")):
        score += 0.08
        reasons.append("title_like_ending")

    # 표 헤더처럼 보이는 숫자/단위가 있으면 감점
    if re.search(r"(당기말|전기말|백만원|천원|억원|%)", norm):
        score -= 0.18
        reasons.append("tableish_numeric_header_penalty")
    if re.search(r"[\d,]{4,}", norm):
        score -= 0.15
        reasons.append("too_many_digits_penalty")
    if len(norm) <= 2:
        score -= 0.15
        reasons.append("too_short_penalty")
    if len(norm) > 80:
        score -= 0.12
        reasons.append("too_long_penalty")

    if prev_text and normalize_inline_text(prev_text) == "주석":
        score += 0.10
        reasons.append("follows_notes")
    if next_text:
        next_norm = normalize_inline_text(next_text)
        if len(next_norm) > len(norm) + 10 and not is_noise_text(next_norm):
            score += 0.05
            reasons.append("followed_by_body")
    if is_noise_text(norm):
        score -= 0.35
        reasons.append("noise_penalty")

    # 패턴 전이 보너스
    if prev_pattern_type != "none":
        if pattern_type == prev_pattern_type:
            score += 0.03
            reasons.append("same_pattern_transition")
        elif pattern_type.startswith("decimal") and prev_pattern_type.startswith("decimal"):
            score += 0.06
            reasons.append("decimal_family_transition")
        elif pattern_type in {"korean_alpha", "paren_num", "circled_num"} and prev_pattern_type.startswith("decimal"):
            score += 0.05
            reasons.append("detail_transition")

    score = max(0.0, min(1.0, score))
    is_real = score >= 0.58
    return score, "|".join(reasons), is_real


# =========================================================
# 6. HTML → 세그먼트 분리
# =========================================================

def extract_text_with_newlines(tag: Tag) -> str:
    """
    태그 내부 텍스트를 추출한다.
    span 분절(PDF 변환 오류)을 merge_inline_spans 로 먼저 재조립한 뒤,
    <br> 태그는 줄바꿈으로 보존한다.
    """
    # span 분절 재조립을 포함한 전체 텍스트 추출
    text = merge_inline_spans(tag)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n", text)
    return normalize_text(text)


def merge_split_numbering_segments(raw_segments: List[str]) -> List[str]:
    """
    번호가 두 줄로 쪼개진 세그먼트를 합친다.
    예: ["2.", "5 현금및현금성자산"] → ["2.5 현금및현금성자산"]
    (decimal_2 패턴이 span 오류로 줄바꿈 되었을 때)
    """
    merged: List[str] = []
    i = 0
    while i < len(raw_segments):
        curr = normalize_inline_text(raw_segments[i])

        if i + 1 < len(raw_segments):
            nxt = normalize_inline_text(raw_segments[i + 1])

            # "숫자." 과 "숫자 내용" 패턴이 연속하면 병합
            m1 = re.match(r"^(\d+)\.$", curr)
            m2 = re.match(r"^(\d+)\.\s+(.+)$", nxt)
            if m1 and m2:
                combined = f"{m1.group(1)}.{m2.group(1)} {m2.group(2)}"
                merged.append(combined)
                i += 2
                continue

            # "숫자" 만 있고 다음 줄이 ".숫자 내용" 형태
            m3 = re.match(r"^(\d+)$", curr)
            m4 = re.match(r"^\.(\d+)\s+(.+)$", nxt)
            if m3 and m4:
                combined = f"{m3.group(1)}.{m4.group(1)} {m4.group(2)}"
                merged.append(combined)
                i += 2
                continue

        merged.append(curr)
        i += 1

    return [x for x in merged if x]


def split_embedded_headers(segment: str) -> List[str]:
    """
    하나의 세그먼트 안에 여러 헤더가 embedded 된 경우 분리한다.
    예: "1. 보고기업 2. 재무제표 작성기준" → 두 개로 분리
    """
    text = normalize_text(segment)
    if not text:
        return []

    lines = [normalize_inline_text(x) for x in text.split("\n") if normalize_inline_text(x)]
    if not lines:
        return []

    out: List[str] = []
    embedded_patterns = [
        r"(?=(?:^|\s)(\d+\.\d+\.\d+[\.\)]?\s+[^\n]+))",
        r"(?=(?:^|\s)(\d+\.\d+[\.\)]?\s+[^\n]+))",
        r"(?=(?:^|\s)(\d+[\.\)]\s+[^\n]+))",
        r"(?=(?:^|\s)(\(\d+\)\s+[^\n]+))",
        r"(?=(?:^|\s)([가-하]\.\s+[^\n]+))",
    ]

    for line in lines:
        splits = [line]
        for pat in embedded_patterns:
            new_splits: List[str] = []
            for item in splits:
                item = normalize_inline_text(item)
                if not item:
                    continue
                starts_like_header = any([
                    re.match(r"^\d+\.\d+\.\d+[\.\)]?\s+", item),
                    re.match(r"^\d+\.\d+[\.\)]?\s+", item),
                    re.match(r"^\d+[\.\)]\s+", item),
                    re.match(r"^\(\d+\)\s+", item),
                    re.match(r"^[가-하]\.\s+", item),
                ])
                if starts_like_header:
                    new_splits.append(item)
                    continue

                matches = list(re.finditer(pat, item))
                valid_positions = []
                for m in matches:
                    pos = m.start(1) if m.lastindex else m.start()
                    if pos > 0:
                        valid_positions.append(pos)

                if not valid_positions:
                    new_splits.append(item)
                    continue

                pieces = []
                prev_pos = 0
                for pos in valid_positions:
                    piece = normalize_inline_text(item[prev_pos:pos])
                    if piece:
                        pieces.append(piece)
                    prev_pos = pos
                tail = normalize_inline_text(item[prev_pos:])
                if tail:
                    pieces.append(tail)
                new_splits.extend(pieces)

            splits = new_splits
        out.extend(splits)

    return [x for x in out if x]


def extract_segments_from_html(htm_path: Path) -> List[Segment]:
    """
    HTML 파일을 읽어 세그먼트 리스트를 반환한다.

    개선점:
    - euc-kr → cp949 → utf-8 순서로 인코딩 fallback
    - lxml → html.parser 순서로 parser fallback
    - p / h1~h6 뿐 아니라 직접 자식 span 이 분절된 경우도 merge_inline_spans 를 통해 복구
    """
    html = _load_html(htm_path)
    soup = _get_soup(html)
    body = soup.body if soup.body else soup

    # 세그먼트를 추출할 블록 태그 (span 은 여기서 직접 처리하지 않음 → merge_inline_spans 에서 처리됨)
    allowed_tags = {"p", "h1", "h2", "h3", "h4", "h5", "h6"}

    segments: List[Segment] = []
    block_index = 0
    segment_id = 0
    prev_text = ""

    for tag in body.find_all(list(allowed_tags)):
        raw = extract_text_with_newlines(tag)
        if not raw:
            block_index += 1
            continue

        line_segments = [normalize_inline_text(x) for x in raw.split("\n") if normalize_inline_text(x)]
        line_segments = merge_split_numbering_segments(line_segments)

        final_segments: List[str] = []
        for seg in line_segments:
            final_segments.extend(split_embedded_headers(seg))

        for seg in final_segments:
            seg = normalize_inline_text(seg)
            if not seg or seg == prev_text:
                continue

            segments.append(Segment(
                segment_id=segment_id,
                source_tag=tag.name,
                block_index=block_index,
                text=seg,
            ))
            prev_text = seg
            segment_id += 1

        block_index += 1

    return segments


# =========================================================
# 7. 후보 추출
# =========================================================

def extract_candidates(segments: List[Segment], threshold: float = 0.45) -> List[HeaderCandidate]:
    """
    세그먼트 목록에서 헤더 후보를 추출한다.
    title_score < threshold 이거나 section_score 가 너무 낮으면 제거.
    """
    candidates: List[HeaderCandidate] = []

    for i, seg in enumerate(segments):
        text = normalize_inline_text(seg.text)
        if not text:
            continue
        if is_continuation_header(text):
            continue
        if is_probable_non_header(text):
            continue

        pattern_type, raw_prefix = classify_pattern(text)
        prev_text = segments[i - 1].text if i > 0 else ""
        next_text = segments[i + 1].text if i + 1 < len(segments) else ""
        prev_pattern_type = classify_pattern(prev_text)[0] if prev_text else "none"

        title_score = compute_title_score(text, pattern_type, prev_text, next_text)

        if pattern_type == "none" and title_score < 0.72:
            continue
        if title_score < threshold:
            continue

        normalized_text = clean_header_text(text)
        candidate_group = choose_candidate_group(pattern_type, prev_pattern_type)
        section_score, reason_bits, is_real_section = compute_section_authenticity_score(
            text, pattern_type, title_score, prev_text, next_text, prev_pattern_type
        )

        if not is_real_section and section_score < 0.50:
            continue

        candidates.append(HeaderCandidate(
            segment_id=seg.segment_id,
            block_index=seg.block_index,
            text=seg.text,
            normalized_text=normalized_text,
            pattern_type=pattern_type,
            raw_prefix=raw_prefix,
            title_score=title_score,
            context_before=normalize_inline_text(prev_text),
            context_after=normalize_inline_text(next_text),
            candidate_type="header_candidate" if section_score >= 0.65 else "uncertain",
            candidate_group=candidate_group,
            section_score=section_score,
            is_real_section=is_real_section,
            reason=reason_bits,
        ))

    return candidates


# =========================================================
# 8. 문서 패턴 프로파일
# =========================================================

def build_pattern_profile(candidates: List[HeaderCandidate]) -> DocumentPatternProfile:
    """문서 전체의 헤더 패턴 분포와 전이 통계를 집계한다."""
    pattern_counts = Counter()
    transition_counts = Counter()

    for cand in candidates:
        pattern_counts[cand.pattern_type] += 1

    for prev_cand, curr_cand in zip(candidates, candidates[1:]):
        transition_counts[(prev_cand.pattern_type, curr_cand.pattern_type)] += 1

    dominant_transitions: Dict[str, str] = {}
    temp: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for (src, dst), count in transition_counts.items():
        temp[src].append((dst, count))
    for src, items in temp.items():
        items.sort(key=lambda x: x[1], reverse=True)
        dominant_transitions[src] = items[0][0]

    ordered = sorted(
        pattern_counts.keys(),
        key=lambda p: (PATTERN_BASE_DEPTH_HINT.get(p, 999), -pattern_counts[p])
    )
    pattern_order_hint = {pt: lvl for lvl, pt in enumerate(ordered, start=1)}

    return DocumentPatternProfile(
        pattern_counts=dict(pattern_counts),
        transition_counts=dict(transition_counts),
        dominant_transitions=dominant_transitions,
        pattern_order_hint=pattern_order_hint,
    )


# =========================================================
# 9. 계층 추론
# =========================================================

def is_same_family(a: str, b: str) -> bool:
    """두 패턴이 같은 계열(decimal / alpha)인지 확인"""
    if a == b:
        return True
    decimal_family = {"decimal_1", "decimal_2", "decimal_3", "note_numeric"}
    alpha_family   = {"korean_alpha", "upper_alpha", "roman", "circled_num", "paren_num"}
    if a in decimal_family and b in decimal_family:
        return True
    if a in alpha_family and b in alpha_family:
        return True
    return False


def get_candidate_depth_options(
    candidate: HeaderCandidate,
    prev_candidate: Optional[HeaderCandidate],
    profile: DocumentPatternProfile,
) -> List[int]:
    """후보에 대해 가능한 깊이 옵션 목록을 반환한다."""
    group_depth = GROUP_TO_DEPTH.get(candidate.candidate_group, 99)
    base = PATTERN_BASE_DEPTH_HINT.get(candidate.pattern_type, 2)
    options = {base}
    if group_depth != 99:
        options.add(group_depth)

    if prev_candidate:
        prev_depth = prev_candidate.inferred_depth or PATTERN_BASE_DEPTH_HINT.get(prev_candidate.pattern_type, 1)

        if candidate.pattern_type == prev_candidate.pattern_type:
            options.add(prev_depth)
        if is_same_family(candidate.pattern_type, prev_candidate.pattern_type):
            options.add(prev_depth)
            options.add(prev_depth + 1)
            if prev_depth > 1:
                options.add(prev_depth - 1)

        if candidate.pattern_type == "decimal_2":
            options.update({2, 3})
        elif candidate.pattern_type == "decimal_3":
            options.update({3, 4})
        elif candidate.pattern_type == "decimal_1":
            options.add(2)
        elif candidate.pattern_type == "korean_alpha":
            options.update({3, 4})
        elif candidate.pattern_type == "paren_num":
            options.update({4, 5})

    return sorted(d for d in options if d >= 1)


def score_depth_choice(
    candidate: HeaderCandidate,
    chosen_depth: int,
    prev_candidate: Optional[HeaderCandidate],
    stack: List[HeaderCandidate],
) -> float:
    """특정 깊이를 선택했을 때의 점수를 계산한다."""
    score = candidate.section_score if candidate.section_score else candidate.title_score

    base_hint = PATTERN_BASE_DEPTH_HINT.get(candidate.pattern_type, 2)
    if chosen_depth == base_hint:
        score += 0.20
    else:
        score -= 0.08 * abs(chosen_depth - base_hint)

    group_depth = GROUP_TO_DEPTH.get(candidate.candidate_group, 99)
    if group_depth != 99:
        if chosen_depth == group_depth:
            score += 0.12
        else:
            score -= 0.05 * abs(chosen_depth - group_depth)

    if prev_candidate:
        prev_depth = prev_candidate.inferred_depth or 1
        if candidate.pattern_type == prev_candidate.pattern_type:
            if chosen_depth == prev_depth:
                score += 0.18
            else:
                score -= 0.08
        if candidate.pattern_type.startswith("decimal") and prev_candidate.pattern_type.startswith("decimal"):
            curr_dots = candidate.raw_prefix.count(".")
            prev_dots = prev_candidate.raw_prefix.count(".")
            delta = curr_dots - prev_dots
            if delta == 0 and chosen_depth == prev_depth:
                score += 0.15
            elif delta == 1 and chosen_depth == prev_depth + 1:
                score += 0.18
            elif delta < 0 and chosen_depth <= prev_depth:
                score += 0.10
        if chosen_depth > prev_depth + 1:
            score -= 0.25

    if chosen_depth > 1 and len(stack) >= chosen_depth - 1:
        score += 0.05

    return score


def infer_hierarchy(
    candidates: List[HeaderCandidate],
    profile: DocumentPatternProfile,
) -> List[InferredNode]:
    """모든 후보에 대해 계층(깊이·부모)을 추론한다."""
    results: List[InferredNode] = []
    stack: List[HeaderCandidate] = []

    for idx, candidate in enumerate(candidates):
        prev_candidate = candidates[idx - 1] if idx > 0 else None
        depth_options = get_candidate_depth_options(candidate, prev_candidate, profile)

        best_depth = None
        best_score = float("-inf")
        for depth in depth_options:
            s = score_depth_choice(candidate, depth, prev_candidate, stack)
            if s > best_score:
                best_score = s
                best_depth = depth

        candidate.inferred_depth = best_depth

        if best_depth is None:
            candidate.status = "suspect"
            candidate.reason = (candidate.reason + " | depth_inference_failed").strip(" |")
            candidate.parent_segment_id = None
        else:
            while len(stack) >= best_depth:
                stack.pop()

            parent = stack[-1] if stack else None
            candidate.parent_segment_id = parent.segment_id if parent else None

            if best_score >= 0.78:
                candidate.status = "confirmed"
            elif best_score >= 0.60:
                candidate.status = "fallback"
            else:
                candidate.status = "suspect"
            candidate.reason = (candidate.reason + f" | best_score={best_score:.2f}").strip(" |")

            if candidate.is_real_section:
                stack.append(candidate)

        results.append(InferredNode(
            segment_id=candidate.segment_id,
            block_index=candidate.block_index,
            text=candidate.normalized_text,
            pattern_type=candidate.pattern_type,
            depth=candidate.inferred_depth,
            parent_segment_id=candidate.parent_segment_id,
            status=candidate.status,
            reason=candidate.reason,
        ))

    return results


# =========================================================
# 10. 후검증 / fallback / continuation 제거
# =========================================================

def validate_and_apply_fallback(
    inferred_nodes: List[InferredNode],
    candidates: List[HeaderCandidate],
) -> List[InferredNode]:
    """
    추론된 계층을 후검증한다.
    - continuation 헤더 제거
    - 깊이 점프 완화
    - 부모 없는 심층 노드 평탄화
    - 점수 낮은 후보 demote
    """
    by_segment_id = {c.segment_id: c for c in candidates}
    fixed: List[InferredNode] = []
    prev_node: Optional[InferredNode] = None

    for node in inferred_nodes:
        new_node = InferredNode(
            segment_id=node.segment_id,
            block_index=node.block_index,
            text=node.text,
            pattern_type=node.pattern_type,
            depth=node.depth,
            parent_segment_id=node.parent_segment_id,
            status=node.status,
            reason=node.reason,
        )

        if is_continuation_header(new_node.text):
            new_node.status = "removed_continuation"
            new_node.reason += " | continuation_removed_post"
            fixed.append(new_node)
            continue

        if new_node.depth is None:
            new_node.status = "suspect"
            new_node.reason += " | no_depth"
            fixed.append(new_node)
            prev_node = new_node
            continue

        if prev_node and prev_node.depth is not None:
            if new_node.depth > prev_node.depth + 1:
                new_node.depth = prev_node.depth + 1
                new_node.status = "fallback"
                new_node.reason += " | depth_jump_smoothed"

        if new_node.depth > 1 and new_node.parent_segment_id is None:
            new_node.depth = 1
            new_node.status = "fallback"
            new_node.reason += " | parent_missing_flattened"

        cand = by_segment_id.get(new_node.segment_id)
        if cand:
            if not cand.is_real_section and new_node.status == "suspect":
                new_node.status = "demoted"
                new_node.reason += " | demoted_fake_section"
            elif cand.title_score < 0.50 and new_node.status == "suspect":
                new_node.status = "demoted"
                new_node.reason += " | demoted_to_body_candidate"

        fixed.append(new_node)
        prev_node = new_node

    return fixed


# =========================================================
# 11. 숫자 파싱 및 단위 추출
# =========================================================

def _is_amount_comma(s: str) -> bool:
    """
    문자열의 쉼표가 '금액 천단위 구분자' 용도인지 판별한다.

    판별 규칙:
    - 쉼표를 제거한 나머지가 순수 숫자(+소수점 허용)이어야 함
    - 첫 번째 쉼표 앞 숫자는 1~3자리 (예: 1,234 → 앞 1자리 OK / 12,34 → 앞 2자리 OK)
    - 이후 쉼표 사이는 정확히 3자리
    - 예) "1,234,567" → True
          "4,6,31"   → False  ← 주석 번호 목록 (각 구간이 3자리 아님)
          "12,5"     → False  ← 소수처럼 보이는 2자리 구간
    """
    # 괄호 벗기기 (음수 표기)
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]

    # 쉼표가 없으면 이 함수로 판별할 필요 없음
    if "," not in s:
        return True  # 쉼표 없는 숫자는 그냥 통과

    parts = s.split(",")

    # 첫 조각: 1~3자리 숫자
    if not re.fullmatch(r"\d{1,3}", parts[0]):
        return False

    # 나머지 조각: 정확히 3자리 숫자
    for part in parts[1:]:
        if not re.fullmatch(r"\d{3}", part):
            return False

    return True


def parse_numeric(text: Any) -> Optional[float]:
    """
    재무제표 수치 문자열 → float 변환.

    처리 규칙:
    - (123,456) → -123456.0   (괄호 = 음수, 회계 관행)
    - 1,234,567 → 1234567.0   (쉼표가 천단위 패턴일 때만)
    - - / – / — → 0.0         (대시 = 0)
    - 빈 문자열  → None
    - 변환 불가  → None        (%, 주 등 비금액 셀)

    ※ 쉼표 오파싱 방지:
       "4, 6, 31" 같은 주석 번호 목록은 쉼표 구간이 3자리가 아니므로
       _is_amount_comma() 검사에서 False → None 반환 (숫자로 합산하지 않음)
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None

    s = str(text).strip().replace("\xa0", "").replace(" ", "")
    if s == "":
        return None
    if s in ("-", "–", "—"):
        return 0.0

    is_negative = s.startswith("(") and s.endswith(")")
    if is_negative:
        s_inner = s[1:-1]
    else:
        s_inner = s

    # 쉼표가 있으면 천단위 패턴인지 먼저 검증
    if "," in s_inner:
        if not _is_amount_comma(s_inner):
            # 쉼표가 있지만 천단위 패턴이 아님 → 숫자로 해석 불가
            return None

    s_clean = s_inner.replace(",", "")

    try:
        value = float(s_clean)
        return -value if is_negative else value
    except ValueError:
        return None


def classify_col_type(header: str, values: List[Any]) -> str:
    """
    컬럼 헤더와 값 샘플로 컬럼 유형을 분류한다.
    반환값: '금액' / '%' / '주' / '주석' / '혼용' / '텍스트'

    - '%' 가 헤더에 있거나 값의 50% 이상이 '%' 포함 → '%'
    - '주석' 이 헤더에 있으면 → '주석' (문자열 그대로 보존, float 변환 안 함)
    - '주' 가 헤더에 있거나 값이 정수처럼 보이고 건수가 작음 → '주'
    - 대부분 float 변환 성공 → '금액'
    - 일부만 변환 성공 → '혼용'
    - 거의 변환 안 됨  → '텍스트'

    ※ '주석' 컬럼: "4, 6, 31" 처럼 쉼표로 구분된 주석 번호 목록.
       헤더에 '주석' 키워드가 있으면 문자열 보존으로 확정한다.
    """
    h = str(header)

    # 주석 컬럼 판단 (헤더에 '주석' 키워드)
    if re.search(r"주\s*석", h):
        return "주석"

    # % 컬럼 판단
    if "%" in h:
        return "%"
    pct_count = sum(1 for v in values if isinstance(v, str) and "%" in v)
    if values and pct_count / len(values) > 0.4:
        return "%"

    # 주(株) 컬럼 판단
    if re.search(r"주\s*수|주식\s*수|발행주식|자기주식", h):
        return "주"

    numeric_count = sum(1 for v in values if parse_numeric(v) is not None)
    total = len([v for v in values if v is not None and str(v).strip() not in ("", "nan")])

    if total == 0:
        return "텍스트"

    ratio = numeric_count / total
    if ratio >= 0.80:
        return "금액"
    elif ratio >= 0.40:
        return "혼용"
    else:
        return "텍스트"


def extract_unit_label(text: str) -> Optional[str]:
    """
    텍스트에서 단위 문자열을 추출한다.
    예: "(단위 : 백만원)" → "백만원"
        "단위: 천원" → "천원"
    """
    text = str(text).replace("\xa0", " ")
    m = re.search(r'\(\s*단위\s*[:\uff1a]\s*([^)]+?)\s*\)', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'단위\s*[:\uff1a]\s*(\S+)', text)
    if m:
        return m.group(1).strip()
    return None


def extract_unit_from_context(table_tag: Tag) -> Optional[str]:
    """
    테이블 태그 바로 앞의 형제 태그들을 최대 10개까지 역방향 탐색하여
    단위 표기를 찾는다. 없으면 테이블 자체 텍스트에서 찾는다.

    참고: 단위 scale_factor 는 저장 시점에 적용하지 않는다.
          분석/질의 시점에 tool calling 등으로 처리할 것.
          단위 예시 → scale_factor 힌트:
            "백만원" → 1_000_000
            "천원"   → 1_000
            "억원"   → 100_000_000
            "%"      → 0.01
    """
    node = table_tag.previous_sibling
    checked = 0
    while node and checked < 10:
        if isinstance(node, Tag):
            unit = extract_unit_label(node.get_text(" ", strip=True))
            if unit:
                return unit
            checked += 1
        node = node.previous_sibling

    return extract_unit_label(table_tag.get_text(" ", strip=True))


# =========================================================
# 12. HTML 파일 IO
# =========================================================

def _load_html(htm_path: Path) -> str:
    """
    한국 감사보고서의 다양한 인코딩에 대응하는 fallback 로더.
    euc-kr → cp949 → utf-8 순으로 시도하고, 모두 실패하면 강제 복구(replace).
    """
    for enc in ("euc-kr", "cp949", "utf-8"):
        try:
            with open(htm_path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue

    # 최후 수단: 깨진 글자는 대체 문자로 치환
    with open(htm_path, "r", encoding="euc-kr", errors="replace") as f:
        return f.read()


def _get_soup(html: str) -> BeautifulSoup:
    """
    parser fallback: lxml → html.parser 순서로 시도.
    lxml 이 설치되어 있으면 더 관대하게 파싱한다.
    """
    for parser in ("lxml", "html.parser"):
        try:
            return BeautifulSoup(html, parser)
        except Exception:
            continue
    return BeautifulSoup(html, "html.parser")


# =========================================================
# 13. 테이블 파싱 및 저장
# =========================================================

def manual_parse_table(table_tag: Tag) -> Optional[pd.DataFrame]:
    """
    pd.read_html 실패 시 직접 thead/tbody 를 순회하여 DataFrame 을 생성한다.

    처리:
    - thead > tr > th/td 를 헤더 행으로 사용
    - tbody > tr > td/th 를 데이터 행으로 사용
    - colspan/rowspan 은 반복 채움(forward-fill)으로 처리
    - thead 가 없으면 첫 번째 tr 을 헤더로 사용
    """
    def expand_row(tr: Tag) -> List[str]:
        """tr 내의 td/th 텍스트를 colspan 만큼 반복하여 반환"""
        cells = []
        for cell in tr.find_all(["td", "th"]):
            text = normalize_inline_text(cell.get_text(" ", strip=True))
            try:
                span = int(cell.get("colspan", 1))
            except (ValueError, TypeError):
                span = 1
            cells.extend([text] * max(span, 1))
        return cells

    all_rows: List[List[str]] = []

    thead = table_tag.find("thead")
    tbody = table_tag.find("tbody")

    # thead 가 있으면 헤더 행 수집
    header_rows: List[List[str]] = []
    if thead:
        for tr in thead.find_all("tr"):
            header_rows.append(expand_row(tr))

    # tbody (또는 thead 없을 때 전체 tr) 데이터 행 수집
    data_source = tbody if tbody else table_tag
    data_rows: List[List[str]] = []
    for tr in data_source.find_all("tr"):
        data_rows.append(expand_row(tr))

    # thead 가 없으면 첫 번째 데이터 행을 헤더로 사용
    if not header_rows and data_rows:
        header_rows = [data_rows[0]]
        data_rows = data_rows[1:]

    if not header_rows:
        return None

    # 멀티 헤더를 " | " 로 합쳐 단일 헤더 생성
    n_cols = max(len(r) for r in header_rows + data_rows) if (header_rows + data_rows) else 0
    if n_cols == 0:
        return None

    # 각 헤더 행을 n_cols 로 패딩
    padded_headers = [r + [""] * (n_cols - len(r)) for r in header_rows]
    columns: List[str] = []
    for col_idx in range(n_cols):
        parts = [row[col_idx] for row in padded_headers if col_idx < len(row) and row[col_idx]]
        # 중복 제거하면서 순서 유지
        seen = []
        for p in parts:
            if p not in seen:
                seen.append(p)
        columns.append(" | ".join(seen) if seen else f"col_{col_idx}")

    # 데이터 행을 n_cols 로 패딩
    padded_data = [r + [""] * (n_cols - len(r)) for r in data_rows]

    if not padded_data:
        return None

    try:
        df = pd.DataFrame(padded_data, columns=columns)
        return df
    except Exception:
        return None


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex 컬럼을 " | " 구분 단일 컬럼명으로 평탄화한다.

    ※ pandas 가 중복 컬럼명에 자동으로 붙이는 '.1', '.2' suffix 를 제거한다.
       예) '제 50(당) 기.1' → '제 50(당) 기 (전기)'  처럼 suffix 를 의미있는 구분자로 교체한다.
       구체적으로: 같은 베이스명이 여러 번 나타나면 두 번째부터 _2, _3, ... 대신
       컬럼 원래 텍스트에서 '.숫자' 접미사만 제거하여 보존하고,
       이후 컬럼명 중복은 col_N 으로 대체한다.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        new_cols: List[str] = []
        for i, col in enumerate(df.columns):
            parts = [str(x).strip() for x in col if str(x).strip() and str(x).strip().lower() != "nan"]
            if not parts:
                new_cols.append(f"col_{i}")
            else:
                dedup = []
                for p in parts:
                    if not dedup or dedup[-1] != p:
                        dedup.append(p)
                new_cols.append(" | ".join(dedup))
        df.columns = new_cols
    else:
        new_cols = []
        for i, col in enumerate(df.columns):
            s = str(col).strip()
            if s in ("nan", "", "None"):
                s = f"col_{i}"
            else:
                # pandas 가 자동으로 붙인 '.숫자' suffix 제거
                # 예: '제 50(당) 기.1' → '제 50(당) 기'
                s = re.sub(r"\.\d+$", "", s).strip()
            new_cols.append(s)
        # suffix 제거 후 중복이 남으면 두 번째부터 _2, _3 으로 재구분
        seen_count: Dict[str, int] = {}
        deduped: List[str] = []
        for s in new_cols:
            if s not in seen_count:
                seen_count[s] = 0
                deduped.append(s)
            else:
                seen_count[s] += 1
                deduped.append(f"{s}_{seen_count[s] + 1}")
        df.columns = deduped
    return df


def clean_financial_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    재무제표 DataFrame 정제.

    전략:
    - 첫 번째 컬럼 = 계정과목명 → 문자열 보존
    - 나머지 컬럼 = parse_numeric() 적용 → float 또는 None
      · 퍼센트, 주수 등 변환 불가 셀은 None 이 아니라 원본 문자열 보존
        (classify_col_type 에서 '혼용' 처리)

    저장 형식:
    - float 컬럼은 float 그대로 CSV 에 저장 (소수점 없는 정수도 .0 형식)
    - 금액 표시가 필요할 때는 별도로 unit 을 참고하여 포매팅
    - 계산(합계, 비율 등)은 float 컬럼을 그대로 사용하면 됨
    """
    if df is None or df.empty:
        return None

    df = flatten_columns(df)

    for col in df.columns[1:]:
        raw_values = df[col].tolist()
        col_type = classify_col_type(col, raw_values)

        if col_type in ("금액", "주"):
            # float 변환 적용 (변환 실패 셀은 None/NaN)
            df[col] = df[col].apply(parse_numeric)
        elif col_type == "주석":
            # 주석 컬럼: 문자열 그대로 보존 (쉼표 구분 주석 번호 목록)
            df[col] = df[col].apply(lambda v: str(v).strip() if v is not None and str(v).strip() not in ("", "nan") else None)
        elif col_type == "%":
            # 퍼센트 컬럼: '%' 제거 후 float 변환, 실패 시 None
            def parse_pct(v: Any) -> Optional[float]:
                if v is None:
                    return None
                s = str(v).replace("%", "").strip()
                try:
                    return float(s.replace(",", ""))
                except ValueError:
                    return None
            df[col] = df[col].apply(parse_pct)
        elif col_type == "혼용":
            # 혼용: 변환 가능한 셀만 float, 나머지는 원본 문자열 유지
            def parse_mixed(v: Any) -> Any:
                result = parse_numeric(v)
                return result if result is not None else (str(v) if v is not None else None)
            df[col] = df[col].apply(parse_mixed)
        # '텍스트' 컬럼은 변환하지 않음

    return df.reset_index(drop=True)


def extract_col_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    정제된 DataFrame 의 각 컬럼 유형을 반환한다.
    (clean_financial_df 적용 이전의 원본 df 를 받아야 정확하지만,
     헤더명 기반으로 사후 분류해도 대부분 정확함)
    """
    col_types = {}
    for i, col in enumerate(df.columns):
        if i == 0:
            col_types[col] = "텍스트"  # 첫 번째 = 계정과목명
        else:
            values = df[col].tolist()
            col_types[col] = classify_col_type(col, values)
    return col_types


def is_text_box_table(table_tag: Tag) -> bool:
    """
    1행×1열 또는 1행×1열 실질 내용인 텍스트 박스 테이블 여부 판단.
    이런 테이블은 CSV 저장 없이 MD 본문에 텍스트로 삽입한다.
    """
    rows = table_tag.find_all("tr")
    if not rows:
        return False

    # 실질적으로 내용이 있는 td/th 셀 수 계산
    total_cells_with_content = 0
    for tr in rows:
        for cell in tr.find_all(["td", "th"]):
            text = normalize_inline_text(cell.get_text(" ", strip=True))
            if text:
                total_cells_with_content += 1

    # 내용 있는 셀이 1개뿐 → 텍스트 박스
    return total_cells_with_content <= 1


def extract_tables_from_html(htm_path: Path, output_dir: Path) -> Tuple[List[TableExportInfo], List[str]]:
    """
    HTML 파일에서 class="TABLE" 인 테이블만 추출하여 저장한다.

    반환: (manifest 목록, 텍스트박스 문장 목록)

    필터링:
    - class="TABLE" 만 처리 (class="nb" 등 제외)
    - 1×1 텍스트 박스 테이블은 CSV 저장 안 하고 텍스트 목록에 추가
    - 빈 DataFrame 은 건너뜀

    저장:
    - tables/table_NNNN.csv  (float 수치 포함)
    - table_manifest.csv     (경로·단위·컬럼유형 메타)
    """
    html = _load_html(htm_path)
    soup = _get_soup(html)

    # class="TABLE" 만 필터 (대소문자 구분 없이)
    tables = soup.find_all("table", class_="TABLE")

    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[TableExportInfo] = []
    text_boxes: List[str] = []   # 1×1 텍스트박스에서 추출한 문장들

    for idx, table_tag in enumerate(tables, start=1):

        # ── 1×1 텍스트 박스 처리 ──────────────────────────────
        if is_text_box_table(table_tag):
            text = normalize_inline_text(table_tag.get_text(" ", strip=True))
            if text:
                text_boxes.append(text)
            continue

        # ── 일반 테이블 파싱 ──────────────────────────────────
        df_raw: Optional[pd.DataFrame] = None

        try:
            dfs = pd.read_html(str(table_tag), flavor="lxml")
            if dfs:
                df_raw = dfs[0]
        except Exception:
            pass

        if df_raw is None:
            # pd.read_html 실패 시 수동 파싱으로 fallback
            df_raw = manual_parse_table(table_tag)

        if df_raw is None or df_raw.empty:
            continue

        # 컬럼 유형 분류 (clean_financial_df 적용 전 원본 기준)
        col_types_raw = extract_col_types(flatten_columns(df_raw.copy()))

        # 수치 변환 적용
        df_clean = clean_financial_df(df_raw)
        if df_clean is None or df_clean.empty:
            continue

        unit = extract_unit_from_context(table_tag)

        csv_name = f"table_{idx:04d}.csv"
        csv_path = table_dir / csv_name
        df_clean.to_csv(csv_path, index=False, encoding="utf-8-sig")

        manifest.append(TableExportInfo(
            table_index=idx,
            csv_path=str(csv_path.relative_to(output_dir)),
            row_count=int(df_clean.shape[0]),
            col_count=int(df_clean.shape[1]),
            unit=unit,
            col_types=json.dumps(col_types_raw, ensure_ascii=False),
        ))

    return manifest, text_boxes


# =========================================================
# 14. 출력 유틸
# =========================================================

def build_tree_repr(nodes: List[InferredNode]) -> str:
    """계층 트리 텍스트 표현 생성"""
    lines = []
    for node in nodes:
        if node.status == "removed_continuation":
            continue
        depth = node.depth or 1
        indent = "  " * (depth - 1)
        lines.append(
            f"{indent}- [{node.pattern_type}] {node.text} "
            f"(depth={node.depth}, parent={node.parent_segment_id}, status={node.status})"
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: List[dict]) -> None:
    """딕셔너리 목록을 CSV 파일로 저장한다."""
    if not rows:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────
# h 태그 깊이 → Markdown 헤더 접두사 매핑
# ─────────────────────────────────────────────────────────
_H_TAG_TO_PREFIX = {
    "h1": "#",
    "h2": "##",
    "h3": "###",
    "h4": "####",
    "h5": "#####",
    "h6": "######",
}

# SECTION 클래스 이름 → 추가 헤딩 깊이 오프셋
# 예) class="SECTION-1" → depth 1 → "#"
# 예) class="SECTION-2" → depth 2 → "##"
_SECTION_CLASS_RE = re.compile(r"SECTION-(\d+)", re.IGNORECASE)


def _get_heading_prefix(tag: Tag) -> str:
    """
    태그의 헤딩 레벨을 결정하여 Markdown 접두사('#', '##', ...) 를 반환한다.

    우선순위:
    1. h1~h6 시맨틱 태그 → 태그명 기반
    2. class 에 'SECTION-N' 패턴 포함 → N 기반
    3. 해당 없으면 빈 문자열 (일반 단락)
    """
    name = tag.name.lower() if tag.name else ""

    # h1~h6 시맨틱 헤더
    if name in _H_TAG_TO_PREFIX:
        return _H_TAG_TO_PREFIX[name]

    # class="SECTION-1" 등 명시적 섹션 클래스
    classes = tag.get("class", [])
    for cls in classes:
        m = _SECTION_CLASS_RE.search(str(cls))
        if m:
            depth = int(m.group(1))
            prefix = "#" * max(1, min(depth, 6))
            return prefix

    return ""


def _tag_to_md_lines(tag: Tag) -> List[str]:
    """
    단일 블록 태그(p, h1~h6)를 Markdown 줄 목록으로 변환한다.

    처리:
    - merge_inline_spans() 로 span 분절 재조립
    - <br> 은 \n 마커로 보존되어 있으므로 줄 단위로 분리
    - 각 줄이 실질 내용이 있는 경우만 포함
    - 헤더 태그면 각 줄 앞에 heading prefix 추가
    """
    prefix = _get_heading_prefix(tag)
    raw = merge_inline_spans(tag)  # br → \n 이미 보존됨

    result_lines: List[str] = []
    for line in raw.split("\n"):
        line = normalize_inline_text(line)
        if not line:
            continue
        if prefix:
            result_lines.append(f"{prefix} {line}")
        else:
            result_lines.append(line)

    return result_lines


def build_markdown(
    soup: BeautifulSoup,
    manifest: List[TableExportInfo],
    text_boxes: List[str],
) -> str:
    """
    HTML DOM 순서를 그대로 유지하면서 Markdown 을 생성한다.

    처리 규칙:
    ┌────────────────────────────────────────────────────────────┐
    │ DOM 요소 유형          │ Markdown 처리                     │
    ├────────────────────────────────────────────────────────────┤
    │ h1~h6 태그             │ # ~ ###### 헤더로 변환            │
    │ class="SECTION-N" h태그│ depth N 에 맞는 # 헤더로 변환    │
    │ p 태그                 │ 일반 텍스트 (줄별로 출력)         │
    │ <br> 경계              │ 각각 별도 줄 (이어붙이지 않음)   │
    │ class="TABLE" 테이블   │ [TABLE: path  단위: XXX] 인라인   │
    │ class="nb" 테이블      │ CSV 저장 안 함. 내부 p/h/td 텍스트는 추출하여 흐름에 삽입 │
    │ 1×1 텍스트박스 테이블  │ 일반 텍스트로 인라인 삽입         │
    └────────────────────────────────────────────────────────────┘

    ※ 이전 버전의 문제점:
       - 텍스트를 모두 먼저 출력한 뒤 테이블 참조를 뒤에 몰아붙임 → 흐름 붕괴
       - h 태그를 일반 텍스트로 처리 → # 헤더 없음
       - br 경계를 무시하고 하나의 줄로 합침 → 의미 단위 분리 소실
    """
    # manifest 를 table_tag 위치와 매핑하기 위해 인덱스 순서 유지
    # (extract_tables_from_html 의 idx 는 class="TABLE" 테이블 중 1부터 시작)
    manifest_by_idx: Dict[int, TableExportInfo] = {m.table_index: m for m in manifest}

    # class="TABLE" 테이블 목록 (DOM 순서, 1-indexed)
    all_table_tags = soup.find_all("table", class_="TABLE")

    # 텍스트박스 인덱스 추적 (text_boxes 는 class="TABLE" 中 1×1 인 것들)
    # → DOM 탐색 중 텍스트박스를 만나면 순서대로 꺼냄
    tb_queue = list(text_boxes)
    tb_iter_idx = 0  # text_boxes 소비 인덱스

    # TABLE 클래스 테이블의 DOM 내 순서 → manifest 인덱스 역매핑
    # (1×1 텍스트박스는 manifest 에 없으므로 None)
    table_tag_to_manifest: Dict[int, Optional[TableExportInfo]] = {}
    table_dom_order: Dict[Tag, int] = {}
    table_counter = 1
    for tbl in all_table_tags:
        table_dom_order[tbl] = table_counter
        if table_counter in manifest_by_idx:
            table_tag_to_manifest[table_counter] = manifest_by_idx[table_counter]
        else:
            table_tag_to_manifest[table_counter] = None  # 텍스트박스였음
        table_counter += 1

    # DOM을 순서대로 순회하며 Markdown 줄 수집
    lines: List[str] = []
    text_box_consumed = 0  # text_boxes 에서 몇 개 소비했는지

    body = soup.body if soup.body else soup

    # ── 넘버링 패턴 → Markdown 헤더 깊이 매핑 ──────────────────────────
    #
    #   1.    (점)         →  ###   (decimal_1)
    #   1.1               →  ####  (decimal_2)
    #   1.1.1             →  ##### (decimal_3)
    #   (1)  (양쪽 괄호)  →  ##### (paren_num) ← 1. 과 동급 또는 그 아래
    #   가.               →  ##### (korean_alpha)
    #   1)   (오른쪽만)   →  ###### ← (1) 보다 한 단계 더 깊음
    #   ①   (동그라미)   →  ###### ← 1) 와 동급
    #   A.                →  ######
    # ── 넘버링 패턴 → Markdown 헤더 깊이 매핑 ──────────────────────────
    #
    #   가나다.   →  ###   (decimal: 1., 2., ...)
    #   1.1       →  ####
    #   1.1.1     →  #####
    #   (1)       →  #####   ← 양쪽 괄호: 1. 과 같은 레벨 또는 그 아래
    #   가.       →  #####   ← 한글: (1) 과 동급
    #   1)        →  ######  ← 오른쪽 괄호만: (1) 보다 한 단계 더 깊음
    #   ①        →  ######  ← 동그라미: 1) 와 동급 (가장 세부)
    #   A.        →  ######
    #
    # ※ 패턴 순서 중요: 더 구체적인(긴) 패턴을 앞에 배치해야 한다.
    #   예) "^\d+\.\d+\.\d+" 를 "^\d+\.\d+" 보다 먼저 검사해야 1.1.1 이 1.1 로 잘못 매칭되지 않음
    #   예) "^\d+\." 를 "^\d+\)" 보다 먼저 배치하면 "1)" 이 decimal_1 로 잘못 매칭될 수 있으므로
    #       두 패턴을 명시적으로 분리해야 함
    _NUMBERING_RULES: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"^\d+\.\d+\.\d+\.?\s+\S"),       "#####"),   # 1.1.1  → depth 5
        (re.compile(r"^\d+\.\d+\.?\s+\S"),              "####"),   # 1.1    → depth 4
        (re.compile(r"^\d+\.\s+\S"),                    "###"),    # 1.     → depth 3  (점만)
        (re.compile(r"^\(\d+\)\s+\S"),                  "#####"),  # (1)    → depth 5  (양쪽 괄호)
        (re.compile(r"^[가-하]\.\s+\S"),                "#####"),  # 가.    → depth 5
        (re.compile(r"^\d+\)\s+\S"),                    "######"), # 1)     → depth 6  (오른쪽 괄호만)
        (re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*\S"),      "######"), # ①     → depth 6
        (re.compile(r"^[A-Z]\.\s+\S"),                  "######"), # A.     → depth 6
    ]

    def _promote_numbering_to_heading(line: str) -> str:
        """
        p 태그 줄이 목차 넘버링 패턴이면 # 헤더로 승격한다.
        이미 # 로 시작하는 줄(h 태그 변환분)은 그대로 반환한다.

        ※ 순서 중요: 넘버링 패턴 매칭을 is_probable_non_header 보다 먼저 수행한다.
           "1) 회계정책을 적용함에 있어..." 처럼 넘버링이 있어도 종결어미로 끝나면
           is_probable_non_header 가 True 를 반환하여 헤더 승격을 막기 때문.
           넘버링 패턴이 확인되면 내용 길이/어미와 무관하게 헤더로 승격한다.

        ※ '계속' 처리:
           - 넘버링 헤더에 붙은 ", 계속 :" 등 연속 표시는 clean_header_text() 로 제거한다.
           - 제거 후 빈 줄이 되면 (e.g. 순수 "계속" 한 단어) 출력하지 않는다.
        """
        if line.startswith("#"):
            # 이미 헤더로 변환된 줄도 계속 꼬리 제거
            hashes, _, rest = line.partition(" ")
            cleaned = clean_header_text(rest)
            return f"{hashes} {cleaned}" if cleaned else ""
        # 빈 줄은 승격 안 함
        if not line.strip():
            return line
        # 넘버링 패턴 먼저 검사 → 매칭되면 즉시 승격 (is_probable_non_header 무시)
        for pattern, prefix in _NUMBERING_RULES:
            if pattern.match(line):
                cleaned = clean_header_text(line)
                return f"{prefix} {cleaned}" if cleaned else ""
        # 넘버링 없는 줄은 기존 필터 적용
        if is_probable_non_header(line):
            return line
        return line

    def _walk(node: Tag) -> None:
        """
        DOM 트리를 순서대로 순회하며 Markdown 줄을 수집한다.

        - class="TABLE"  → CSV 참조 삽입 (1×1 텍스트박스는 일반 텍스트)
        - class="nb"     → CSV 저장 안 함, 내부 p/h 텍스트는 반드시 추출
                           (재무제표 앞뒤 회사명·기간·서명·별첨주석 등이 nb 안에 있음)
        - h1~h6          → Markdown # 헤더
        - p              → 일반 텍스트, 넘버링 패턴이면 # 헤더 승격
        - td/th/tr 등    → 재귀 탐색 (nb 내부 셀 순회)
        - div/section    → 재귀 탐색 (컨테이너)
        """
        nonlocal text_box_consumed
        for child in node.children:
            if not isinstance(child, Tag):
                continue

            tag_name = child.name.lower() if child.name else ""

            # ── 테이블 ────────────────────────────────────────
            if tag_name == "table":
                classes = child.get("class", [])
                class_str = " ".join(str(c) for c in classes)

                if "TABLE" in class_str:
                    # 재무 데이터 테이블 → CSV 참조 삽입
                    dom_idx = table_dom_order.get(child)
                    if dom_idx is not None:
                        info = table_tag_to_manifest.get(dom_idx)
                        if info is None:
                            # 1×1 텍스트박스
                            if text_box_consumed < len(text_boxes):
                                lines.append(text_boxes[text_box_consumed])
                                text_box_consumed += 1
                        else:
                            unit_str = f"  단위: {info.unit}" if info.unit else ""
                            lines.append(f"[TABLE: {info.csv_path}{unit_str}]")

                elif "nb" in class_str.lower():
                    # 레이아웃용 nb 테이블: CSV 저장 안 하지만 내부 텍스트는 추출
                    _walk(child)

                else:
                    # 기타 테이블: 내부 텍스트 추출
                    _walk(child)

                continue

            # ── h1~h6 시맨틱 헤더 ────────────────────────────
            if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                lines.extend(_tag_to_md_lines(child))
                continue

            # ── p: 일반 텍스트 또는 넘버링 → # 헤더 승격 ────
            if tag_name == "p":
                for tl in _tag_to_md_lines(child):
                    lines.append(_promote_numbering_to_heading(tl))
                continue

            # ── 테이블 내부 구조 (nb 재귀용) ─────────────────
            if tag_name == "tr":
                # tr 안의 td/th 를 검사:
                # - 모든 셀이 단순 텍스트(블록 자식 없음)이면 → 공백으로 합쳐 한 줄 출력
                # - 하나라도 블록 자식(p/h/table)이 있으면 → 기존대로 재귀
                #
                # 이렇게 해야 "0000년 00월 00일" 과 "까지" 가 같은 tr 의
                # 서로 다른 td 에 있을 때 "0000년  00월 00일 00까지" 로 합쳐진다.
                cells = child.find_all(["td", "th"], recursive=False)
                if not cells:
                    # colspan 등으로 셀이 직접 자식이 아닐 수 있음 → 재귀
                    _walk(child)
                    continue

                all_simple = all(
                    not any(
                        isinstance(c, Tag) and c.name in {
                            "p", "h1", "h2", "h3", "h4", "h5", "h6", "table"
                        }
                        for c in cell.children
                    )
                    for cell in cells
                )

                if all_simple:
                    # 단순 텍스트 셀들: 공백으로 합쳐 한 줄
                    parts = []
                    for cell in cells:
                        t = normalize_inline_text(merge_inline_spans(cell))
                        if t:
                            parts.append(t)
                    if parts:
                        lines.append(" ".join(parts))
                else:
                    # 블록 자식 포함 셀 있음: 재귀로 각 셀 처리
                    _walk(child)
                continue

            if tag_name in {"td", "th"}:
                # td/th 가 tr 을 거치지 않고 직접 순회될 때 (thead/tbody 재귀 등)
                has_block_child = any(
                    isinstance(c, Tag) and c.name in {
                        "p", "h1", "h2", "h3", "h4", "h5", "h6", "table"
                    }
                    for c in child.children
                )
                if has_block_child:
                    _walk(child)
                else:
                    text = merge_inline_spans(child)
                    text = normalize_inline_text(text)
                    if text:
                        lines.append(text)
                continue

            if tag_name in {"thead", "tbody", "tfoot"}:
                _walk(child)
                continue

            # ── 컨테이너 태그 재귀 ───────────────────────────
            if tag_name in {"div", "section", "article", "main", "body", "blockquote"}:
                _walk(child)

    _walk(body)

    # 독립 '계속;' 줄 판정 패턴
    # — 앞뒤와 무관하게 "계속;" / "계속 ;" / "계속" 만 달랑 있는 줄 제거
    _standalone_cont = re.compile(r"^\s*계속\s*;?\s*$")

    # 빈 줄 중복 제거 + 독립 '계속;' 줄 제거 후 "\n\n" 으로 결합
    result_lines: List[str] = []
    for line in lines:
        if not line:
            continue
        if _standalone_cont.match(line):
            continue
        result_lines.append(line)

    return "\n\n".join(result_lines)


# =========================================================
# 15. 전체 파이프라인
# =========================================================

def parse_document_structure(htm_path: Path, output_dir: Path) -> Dict[str, Any]:
    """
    단일 HTM 파일의 전체 파싱 파이프라인.

    출력 파일:
    - tables/table_NNNN.csv   (테이블별 float 수치 포함 CSV)
    - table_manifest.csv      (경로·단위·컬럼유형 메타)
    - tree.txt                (트리 텍스트 표현)
    - {파일명}.md             (문서 Markdown)

    제거된 출력 (불필요):
    - segments.csv
    - candidate_list.csv
    - final_hierarchy.csv     (tree.txt 로 충분)
    - summary.md  (summary 내용은 tree.txt + manifest 로 충분)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 섹션 계층 파싱 ────────────────────────────────────
    segments = extract_segments_from_html(htm_path)
    candidates = extract_candidates(segments)
    profile = build_pattern_profile(candidates)
    inferred = infer_hierarchy(candidates, profile)
    final_nodes = validate_and_apply_fallback(inferred, candidates)
    tree = build_tree_repr(final_nodes)


    with open(output_dir / "tree.txt", "w", encoding="utf-8") as f:
        f.write(tree)

    # ── 테이블 추출 ───────────────────────────────────────
    manifest, text_boxes = extract_tables_from_html(htm_path, output_dir)

    # table_manifest.csv: 경로 관리용 (unit, col_types 포함)
    write_csv(output_dir / "table_manifest.csv", [asdict(x) for x in manifest])

    # ── Markdown 생성 ─────────────────────────────────────
    html = _load_html(htm_path)
    soup = _get_soup(html)
    md_content = build_markdown(soup, manifest, text_boxes)
    md_path = output_dir / f"{htm_path.stem}.md"
    md_path.write_text(md_content, encoding="utf-8")

    return {
        "segments": segments,
        "candidates": candidates,
        "profile": profile,
        "inferred_nodes": inferred,
        "final_nodes": final_nodes,
        "tree": tree,
        "table_manifest": manifest,
        "text_boxes": text_boxes,
    }


# =========================================================
# 16. 배치 처리
# =========================================================

def batch(input_dir: str, output_dir: str) -> None:
    """
    입력 디렉터리의 모든 .htm 파일을 순차 처리한다.
    각 파일의 출력은 output_dir/{파일명}/ 하위에 저장된다.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    htm_files = sorted(input_path.glob("*.htm"))
    if not htm_files:
        print(f"[경고] {input_dir} 에 .htm 파일이 없습니다.")
        return

    for htm_file in htm_files:
        print(f"[처리 중] {htm_file.name}")
        file_out = output_path / htm_file.stem
        try:
            result = parse_document_structure(htm_file, file_out)
            n_tables = len(result["table_manifest"])
            n_nodes  = len([n for n in result["final_nodes"] if n.status != "removed_continuation"])
            print(f"  → 섹션 헤더 {n_nodes}개, 테이블 {n_tables}개 추출 완료")
        except Exception as e:
            print(f"  [오류] {htm_file.name}: {e}")


# =========================================================
# 17. 실행 예시
# =========================================================

if __name__ == "__main__":
    # 배치 처리: 삼성전자_감사보고서_2014_2024 폴더의 모든 .htm 파일을 처리
    # 입력: 삼성전자_감사보고서_2014_2024/감사보고서_2014.htm ~ 감사보고서_2024.htm
    # 출력: output/감사보고서_2014/ ~ output/감사보고서_2024/
    #        각 폴더 내부: tables/table_NNNN.csv, table_manifest.csv, tree.txt, 감사보고서_YYYY.md

    
    INPUT_DIR  = Path(__file__).parent.parent / "parsed_data" / "삼성전자_감사보고서_2014_2024"
    OUTPUT_DIR = Path(__file__).parent.parent / "parsed_data" / "output"
    
    batch(str(INPUT_DIR), str(OUTPUT_DIR))
