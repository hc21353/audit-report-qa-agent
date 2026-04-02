# Tool 가이드

## 개요

에이전트가 사용하는 tool은 크게 세 카테고리로 나뉜다.

## 검색 도구 (Retrieval)

### vector_search

벡터 DB에서 의미 유사도 기반으로 문서 청크를 검색한다.

**언제 사용하나**: 개념적 질문, 설명 요청, 정책/방침 관련 질문
```
예: "건설중인자산의 감가상각 정책은?"
    "리스 회계처리 방법이 뭐야?"
```

**필터 옵션**: year, year_from~year_to, section_path

### sql_query

SQLite에서 정형 데이터를 직접 조회한다. 숫자 비교에 핵심.

**언제 사용하나**: 수치 조회, 연도 비교, 계정 검색
```
예: "2024년 총자산" → account_lookup
    "2018 vs 2022 부채" → year_comparison
    "최근 5년 유동자산 추이" → trend
```

**안전장치**: read_only, DROP/DELETE/UPDATE/INSERT/ALTER 차단, 최대 1000행

### section_filter

특정 섹션의 모든 청크를 가져온다.

**언제 사용하나**: "감사의견 전체 보여줘", "주석 30번 내용"
```
예: section_path="독립된 감사인의 감사보고서"
    section_path="주석 > 30. 특수관계자 거래"
```

## 분석 도구 (Analysis)

### calculator

재무비율 계산. 프리셋(유동비율, 부채비율, ROE 등)과 커스텀 수식 지원.

**흐름**: sql_query로 숫자 조회 → calculator로 계산
```
예: "2024년 부채비율"
    → sql_query: 부채총계, 자본총계 조회
    → calculator: operation=ratio, 부채총계/자본총계*100
```

### chart_generator

Plotly 기반 차트 생성. bar, line, stacked_bar, pie, waterfall 지원.

**흐름**: sql_query로 데이터 조회 → chart_generator로 시각화
```
예: "최근 5년 부채비율 추이 차트"
    → sql_query: 5년치 부채총계, 자본총계
    → calculator: 연도별 비율 계산
    → chart_generator: line 차트
```

## 외부 도구 (External / MCP)

### web_search

웹 검색으로 최신 정보를 가져온다. 감사보고서에 없는 시장 맥락 보충용.

**언제 사용하나**: 최신 뉴스, 시장 동향, 보고서 해석에 필요한 외부 정보
```
예: "삼성전자 최근 반도체 실적 뉴스"
    "2024년 K-IFRS 개정 내용"
```

### dart_api (예정)

DART 전자공시시스템 연동. API 키 필요.
