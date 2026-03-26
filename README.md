#  Samsung Electronics Audit Report NLP & RAG System

삼성전자 10개년(2014–2024) 감사보고서 HTML 데이터를 정제하고, 데이터 파이프라인 및 CI/CD 환경을 구축하여 금융 도메인에 특화된 RAG 기반 QA 시스템 개발

---

##  프로젝트 목표
- **E2E 데이터 파이프라인**: 10년치 HTML 감사보고서의 구조적 통합 파싱 및 정제
- **신뢰성 있는 QA 시스템**: 하이브리드 검색(의미+키워드) 및 Tool Calling 기반의 연도별 비교 질의 시스템 구축
- **안정적인 개발 환경**: Git 브랜치 전략, 유닛 테스트(`pytest`), GitHub Actions를 통한 CI/CD 환경 확보
