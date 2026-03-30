import pytest

# 나중에 파싱팀이 만들 함수를 임포트한다고 가정
# from src.parser import html_to_data 

def test_initial_setup():
    """도커와 pytest가 잘 연결되었는지 확인하는 기본 테스트"""
    assert 1 + 1 == 2

def test_parsing_schema_promise():
    """우리 팀이 약속한 JSON 키값이 존재하는지 가상 테스트"""
    # 임시 가짜 데이터 (나중에 실제 함수 결과값으로 대체)
    sample_result = {
        "year": 2024,
        "firm": "Samjong",
        "operating_income": 1000
    }
    
    expected_keys = ["year", "firm", "operating_income"]
    for key in expected_keys:
        assert key in sample_result, f"필수 키값 '{key}'가 결과에 없습니다!"