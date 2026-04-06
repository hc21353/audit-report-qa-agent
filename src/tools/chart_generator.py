"""
chart_generator.py - Plotly 기반 차트 생성 Tool
"""

import json
from langchain_core.tools import tool


@tool
def chart_generator(
    chart_type: str,
    data: str,
    title: str = "",
) -> str:
    """재무 데이터 시각화 차트를 생성합니다.

    Args:
        chart_type: 차트 유형 (bar, line, stacked_bar, pie)
        data: JSON 문자열. 형식: {"labels": [...], "datasets": [{"name": "...", "values": [...]}]}
        title: 차트 제목

    Returns:
        Plotly figure JSON
    """
    try:
        if isinstance(data, str):
            chart_data = json.loads(data)
        else:
            chart_data = data
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid data JSON: {e}"}, ensure_ascii=False)

    labels = chart_data.get("labels", [])
    datasets = chart_data.get("datasets", [])

    if not labels or not datasets:
        return json.dumps({"error": "data must have 'labels' and 'datasets'"}, ensure_ascii=False)

    # Plotly figure 구조 생성
    traces = []
    for ds in datasets:
        trace = {
            "x": labels,
            "y": ds.get("values", []),
            "name": ds.get("name", ""),
        }

        if chart_type == "bar":
            trace["type"] = "bar"
        elif chart_type == "line":
            trace["type"] = "scatter"
            trace["mode"] = "lines+markers"
        elif chart_type == "stacked_bar":
            trace["type"] = "bar"
        elif chart_type == "pie":
            trace["type"] = "pie"
            trace["labels"] = labels
            trace["values"] = ds.get("values", [])
            del trace["x"]
            del trace["y"]

        traces.append(trace)

    layout = {
        "title": {"text": title},
        "xaxis": {"title": ""},
        "yaxis": {"title": ""},
    }

    if chart_type == "stacked_bar":
        layout["barmode"] = "stack"

    figure = {
        "data": traces,
        "layout": layout,
    }

    return json.dumps({
        "success": True,
        "chart_type": chart_type,
        "figure": figure,
    }, ensure_ascii=False)
