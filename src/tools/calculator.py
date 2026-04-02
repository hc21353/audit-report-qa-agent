"""
calculator.py - 재무비율 및 수치 계산 Tool
"""

import json
import math
from langchain_core.tools import tool


# 재무비율 프리셋
PRESETS = {
    "유동비율": "유동자산 / 유동부채 * 100",
    "부채비율": "부채총계 / 자본총계 * 100",
    "자기자본비율": "자본총계 / 자산총계 * 100",
    "영업이익률": "영업이익 / 매출액 * 100",
    "순이익률": "당기순이익 / 매출액 * 100",
    "ROE": "당기순이익 / 자본총계 * 100",
    "ROA": "당기순이익 / 자산총계 * 100",
    "매출채권회전율": "매출액 / 매출채권",
}


@tool
def calculator(expression: str, label: str = "") -> str:
    """수식을 계산합니다. 재무비율, 증감률, 차이 등을 계산할 때 사용하세요.

    Args:
        expression: 계산식 (예: "82320322 / 80157976 * 100", "((324966127 - 296857289) / 296857289) * 100")
        label: 결과에 붙일 레이블 (예: "유동비율", "자산 증가율")

    Returns:
        계산 결과 JSON
    """
    # 안전한 eval을 위해 허용 함수/상수만 제공
    safe_globals = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
    }

    try:
        # 콤마 제거 (숫자에 포함된 경우)
        clean_expr = expression.replace(",", "")
        result = eval(clean_expr, safe_globals)
        rounded = round(float(result), 4)

        output = {
            "success": True,
            "expression": expression,
            "result": rounded,
            "label": label,
        }

        # 프리셋 매칭 (참고 정보)
        if label in PRESETS:
            output["preset_formula"] = PRESETS[label]

        return json.dumps(output, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "expression": expression,
            "error": str(e),
            "hint": f"사용 가능한 프리셋: {list(PRESETS.keys())}",
        }, ensure_ascii=False)
