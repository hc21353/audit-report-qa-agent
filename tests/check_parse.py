"""
parse_final.py 의 핵심 함수(파싱/분류/변환 로직)들에 대한 단위 테스트 모음.
파일 IO나 외부 의존성이 있는 함수들은 의도적으로 제외, 순수 함수들에 집중하여 테스트 구성.

테스트 구성:

  Section 1  normalize_text / normalize_inline_text
  Section 2  is_noise_text / is_probable_non_header / is_continuation_header
  Section 3  merge_inline_spans          (span 분절 재조립)
  Section 4  _is_amount_comma / parse_numeric
  Section 5  classify_col_type
  Section 6  clean_financial_df          (DataFrame 수치 변환)
  Section 7  flatten_columns             (.1 suffix 제거 + 중복 컬럼)
  Section 8  extract_unit_label
  Section 9  merge_split_numbering_segments / split_embedded_headers
  Section 10 classify_pattern
  Section 11 build_markdown              (Markdown 생성 통합 테스트)
               - nb 테이블 텍스트 추출 및 DOM 순서 유지
               - tr 셀 합치기 (같은 행의 td → 한 줄)
               - 넘버링 패턴 → # 헤더 승격
               - TABLE 참조 인라인 삽입
  Section 12 clean_header_text / 계속 필터 (신규)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent / "parsed_data"))

import parse_final as P  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def make_tag(html_fragment: str) -> BeautifulSoup:
    return BeautifulSoup(html_fragment, "html.parser")


def first_tag(html_fragment: str):
    soup = make_tag(html_fragment)
    return next(t for t in soup.children if hasattr(t, "name") and t.name)


def build_md(html_body: str, manifest=None, text_boxes=None) -> str:
    """<body> 안에 들어갈 HTML 조각을 받아 build_markdown 결과를 반환한다."""
    full_html = f"<html><body>{html_body}</body></html>"
    soup = BeautifulSoup(full_html, "html.parser")
    return P.build_markdown(soup, manifest or [], text_boxes or [])


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 : normalize_text / normalize_inline_text
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_nbsp_replaced(self):
        assert P.normalize_inline_text("A\xa0B") == "A B"

    def test_ideographic_space_replaced(self):
        assert P.normalize_inline_text("A\u3000B") == "A B"

    def test_multiple_spaces_collapsed(self):
        assert P.normalize_inline_text("A   B   C") == "A B C"

    def test_newline_collapsed_inline(self):
        assert P.normalize_inline_text("A\nB\nC") == "A B C"

    def test_normalize_text_preserves_newline(self):
        result = P.normalize_text("A\n\n\nB")
        assert "\n" in result
        # 연속 \n 은 하나로 줄어야 한다
        assert "\n\n" not in result

    def test_strip(self):
        assert P.normalize_inline_text("  hello  ") == "hello"

    def test_empty(self):
        assert P.normalize_inline_text("") == ""
        assert P.normalize_inline_text("   ") == ""


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 : is_noise_text / is_probable_non_header / is_continuation_header
# ─────────────────────────────────────────────────────────────────────────────

class TestTextClassifiers:

    # ── is_noise_text ──────────────────────────────────────
    def test_empty_is_noise(self):
        assert P.is_noise_text("") is True
        assert P.is_noise_text("   ") is True

    def test_pure_number_is_noise(self):
        assert P.is_noise_text("123") is True
        assert P.is_noise_text("(123)") is True

    def test_date_only_is_noise(self):
        assert P.is_noise_text("2018년 12월 31일") is True

    def test_meaningful_text_not_noise(self):
        assert P.is_noise_text("1. 보고기업") is False
        assert P.is_noise_text("2.1 재무제표 작성기준") is False

    # ── is_probable_non_header ─────────────────────────────
    def test_sentence_ending_is_non_header(self):
        assert P.is_probable_non_header(
            "회계정책의 적용에 있어 경영진의 판단을 요구하고 있습니다"
        ) is True

    def test_doeseotseumnida_not_in_regex(self):
        # "되었습니다" 는 현재 정규식 범위 밖 → non-header 로 분류 안 됨 (동작 문서화)
        result = P.is_probable_non_header(
            "이 재무제표는 한국채택국제회계기준에 따라 작성되었습니다."
        )
        assert result is False

    def test_short_title_is_header(self):
        assert P.is_probable_non_header("2.1 재무제표 작성기준") is False

    def test_too_long_is_non_header(self):
        long_text = "가. " + "가" * 130
        assert P.is_probable_non_header(long_text) is True

    # ── is_continuation_header ─────────────────────────────
    def test_continuation_colon(self):
        assert P.is_continuation_header("2.1 금융상품, 계속:") is True

    def test_continuation_space_colon(self):
        # "계속 :" 형태 (공백 포함) — 핵심 버그 수정 케이스
        assert P.is_continuation_header("2. 중요한 회계처리방침, 계속 :") is True

    def test_continuation_no_colon(self):
        assert P.is_continuation_header("주석 3, 계속") is True

    def test_continuation_parenthesis(self):
        assert P.is_continuation_header("주석 3 (계속)") is True

    def test_no_continuation_normal(self):
        assert P.is_continuation_header("2.1 재무제표 작성기준") is False

    def test_no_continuation_sentence(self):
        # "계속" 이 문장 중간에 있는 경우 → False
        assert P.is_continuation_header("계속적인 사업을 영위합니다.") is False


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 : merge_inline_spans
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeInlineSpans:

    def test_split_decimal_number(self):
        tag = first_tag("<p><span>2</span><span>.5 현금및현금성자산</span></p>")
        assert P.merge_inline_spans(tag) == "2.5 현금및현금성자산"

    def test_simple_text(self):
        tag = first_tag("<p>재무상태표</p>")
        assert P.merge_inline_spans(tag) == "재무상태표"

    def test_br_becomes_newline(self):
        tag = first_tag("<p>A<br/>B</p>")
        result = P.merge_inline_spans(tag)
        assert "\n" in result
        assert "A" in result and "B" in result

    def test_nbsp_normalized(self):
        tag = first_tag("<p>A\xa0B</p>")
        assert P.merge_inline_spans(tag) == "A B"

    def test_empty_tag(self):
        tag = first_tag("<p></p>")
        assert P.merge_inline_spans(tag) == ""

    def test_multiple_spans_normal_text(self):
        tag = first_tag("<p><span>재무</span><span>상태표</span></p>")
        result = P.merge_inline_spans(tag)
        assert "재무" in result and "상태표" in result


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 : _is_amount_comma / parse_numeric
# ─────────────────────────────────────────────────────────────────────────────

class TestParseNumeric:

    # ── _is_amount_comma ───────────────────────────────────
    def test_valid_thousand_separator(self):
        assert P._is_amount_comma("1,234,567") is True
        assert P._is_amount_comma("34,113,871") is True
        assert P._is_amount_comma("807,262") is True

    def test_note_list_comma_rejected(self):
        assert P._is_amount_comma("4,6,31") is False
        assert P._is_amount_comma("4,6") is False
        assert P._is_amount_comma("5,6,31") is False

    def test_no_comma_passes(self):
        assert P._is_amount_comma("12345") is True

    # ── parse_numeric ──────────────────────────────────────
    def test_plain_integer(self):
        assert P.parse_numeric("12345") == 12345.0

    def test_thousand_comma(self):
        assert P.parse_numeric("2,607,957") == 2607957.0
        assert P.parse_numeric("34,113,871") == 34113871.0

    def test_negative_parenthesis(self):
        assert P.parse_numeric("(1,234)") == -1234.0
        assert P.parse_numeric("(807,262)") == -807262.0

    def test_dash_is_zero(self):
        assert P.parse_numeric("-") == 0.0
        assert P.parse_numeric("–") == 0.0
        assert P.parse_numeric("—") == 0.0

    def test_note_list_returns_none(self):
        assert P.parse_numeric("4, 6, 31") is None
        assert P.parse_numeric("4,6,31") is None
        assert P.parse_numeric("5,6,31") is None

    def test_empty_returns_none(self):
        assert P.parse_numeric("") is None
        assert P.parse_numeric(None) is None

    def test_text_returns_none(self):
        assert P.parse_numeric("현금및현금성자산") is None
        assert P.parse_numeric("N/A") is None

    def test_percent_returns_none(self):
        assert P.parse_numeric("12.5%") is None


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 : classify_col_type
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyColType:

    def test_note_col_by_header(self):
        result = P.classify_col_type("주석", ["4, 6, 31", "5, 6, 31", "6, 10"])
        assert result == "주석"

    def test_note_col_with_spaces_in_header(self):
        result = P.classify_col_type("주  석", ["4,6,31"])
        assert result == "주석"

    def test_amount_col(self):
        values = ["2,607,957", "34,113,871", "24,933,267", "807,262"]
        result = P.classify_col_type("당기", values)
        assert result == "금액"

    def test_percent_col_by_header(self):
        result = P.classify_col_type("지분율(%)", ["50.0%", "30.0%", "20.0%"])
        assert result == "%"

    def test_stock_col_by_header(self):
        result = P.classify_col_type("발행주식수", ["1000000", "500000"])
        assert result == "주"

    def test_text_col(self):
        values = ["현금및현금성자산", "단기금융상품", "매출채권"]
        result = P.classify_col_type("계정과목", values)
        assert result == "텍스트"

    def test_mixed_col(self):
        values = ["1,000", "해당없음", "2,000", "미해당", "3,000"]
        result = P.classify_col_type("금액", values)
        assert result == "혼용"


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 : clean_financial_df
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanFinancialDf:

    def _make_df(self, data: dict) -> pd.DataFrame:
        return pd.DataFrame(data)

    def test_amount_col_converted_to_float(self):
        df = self._make_df({
            "계정과목": ["현금및현금성자산", "단기금융상품"],
            "당기":     ["2,607,957",        "34,113,871"],
        })
        result = P.clean_financial_df(df)
        assert result["당기"].tolist() == [2607957.0, 34113871.0]

    def test_note_col_preserved_as_string(self):
        df = self._make_df({
            "계정과목": ["현금및현금성자산", "단기금융상품"],
            "주석":     ["4, 6, 31",         "5, 6, 31"],
            "당기":     ["2,607,957",         "34,113,871"],
        })
        result = P.clean_financial_df(df)
        assert result["주석"].tolist() == ["4, 6, 31", "5, 6, 31"]
        assert result["당기"].tolist() == [2607957.0, 34113871.0]

    def test_first_col_unchanged(self):
        df = self._make_df({
            "계정과목": ["현금및현금성자산", "1. 유동자산"],
            "당기":     ["100",              "200"],
        })
        result = P.clean_financial_df(df)
        assert result["계정과목"].tolist() == ["현금및현금성자산", "1. 유동자산"]

    def test_negative_parenthesis_converted(self):
        df = self._make_df({
            "계정과목": ["충당부채"],
            "당기":     ["(1,234,567)"],
        })
        result = P.clean_financial_df(df)
        assert result["당기"].tolist() == [-1234567.0]

    def test_dash_converted_to_zero(self):
        df = self._make_df({
            "계정과목": ["미수금"],
            "당기":     ["-"],
        })
        result = P.clean_financial_df(df)
        assert result["당기"].tolist() == [0.0]

    def test_empty_df_returns_none(self):
        result = P.clean_financial_df(pd.DataFrame())
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 : flatten_columns
# ─────────────────────────────────────────────────────────────────────────────

class TestFlattenColumns:

    def test_dot_suffix_removed(self):
        df = pd.DataFrame(columns=["과목", "제 50(당) 기", "제 50(당) 기.1"])
        result = P.flatten_columns(df)
        cols = list(result.columns)
        assert cols[1] == "제 50(당) 기"
        assert cols[2] == "제 50(당) 기_2"

    def test_no_suffix_unchanged(self):
        df = pd.DataFrame(columns=["계정과목", "당기", "전기"])
        result = P.flatten_columns(df)
        assert list(result.columns) == ["계정과목", "당기", "전기"]

    def test_nan_col_renamed(self):
        # 'nan' 컬럼명은 col_N (N = 컬럼 위치 인덱스) 으로 대체된다
        df = pd.DataFrame(columns=["과목", "nan", "당기"])
        result = P.flatten_columns(df)
        assert result.columns[1] == "col_1"

    def test_multiple_duplicate_suffix(self):
        df = pd.DataFrame(columns=["과목", "당기", "당기.1", "당기.2"])
        result = P.flatten_columns(df)
        cols = list(result.columns)
        assert cols[1] == "당기"
        assert cols[2] == "당기_2"
        assert cols[3] == "당기_3"


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 : extract_unit_label
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractUnitLabel:

    def test_parenthesis_unit(self):
        assert P.extract_unit_label("(단위 : 백만원)") == "백만원"

    def test_colon_unit(self):
        assert P.extract_unit_label("단위: 천원") == "천원"

    def test_fullwidth_colon(self):
        assert P.extract_unit_label("(단위 ： 억원)") == "억원"

    def test_no_unit_returns_none(self):
        assert P.extract_unit_label("재무상태표") is None
        assert P.extract_unit_label("") is None

    def test_nbsp_in_unit_text(self):
        assert P.extract_unit_label("(단위\xa0:\xa0백만원)") == "백만원"


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 : merge_split_numbering_segments / split_embedded_headers
# ─────────────────────────────────────────────────────────────────────────────

class TestSegmentProcessing:

    # ── merge_split_numbering_segments ─────────────────────
    def test_merge_split_decimal(self):
        # "숫자." + "숫자. 내용" 형태만 병합 대상
        segs = ["2.", "5. 현금및현금성자산"]
        result = P.merge_split_numbering_segments(segs)
        assert len(result) == 1
        assert result[0] == "2.5 현금및현금성자산"

    def test_no_merge_without_trailing_dot(self):
        # 뒤 세그먼트에 점이 없으면 병합 조건 불충족 → 원본 유지
        segs = ["2.", "5 현금및현금성자산"]
        result = P.merge_split_numbering_segments(segs)
        assert result == ["2.", "5 현금및현금성자산"]

    def test_no_merge_needed(self):
        segs = ["2.1 재무제표 작성기준", "2.2 회계정책과 공시의 변경"]
        result = P.merge_split_numbering_segments(segs)
        assert result == segs

    # ── split_embedded_headers ─────────────────────────────
    def test_split_embedded_single(self):
        result = P.split_embedded_headers("2.1 재무제표 작성기준")
        assert result == ["2.1 재무제표 작성기준"]

    def test_no_split_when_starts_with_header(self):
        # 이미 헤더 패턴으로 시작하는 텍스트는 분리 시도 안 함 (의도된 동작)
        result = P.split_embedded_headers("1. 보고기업 2. 재무제표 작성기준")
        assert len(result) == 1

    def test_split_embedded_two_headers(self):
        # 비헤더 텍스트 앞에 여러 헤더가 embedded 된 경우 분리
        text = "개요 1. 보고기업 2. 재무제표 작성기준"
        result = P.split_embedded_headers(text)
        assert len(result) == 3
        assert any("보고기업" in r for r in result)
        assert any("재무제표 작성기준" in r for r in result)


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 : classify_pattern
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyPattern:

    CASES = [
        ("주석",                  "section_class_1"),
        ("주석 3",                "note_numeric"),
        ("1. 보고기업",           "decimal_1"),
        ("2.1 재무제표 작성기준", "decimal_2"),
        ("2.1.1 세부기준",        "decimal_3"),
        ("(1) 주요 내용",         "paren_num"),
        ("가. 회사 개요",         "korean_alpha"),
        ("① 소항목",             "circled_num"),
        ("A. 기타사항",           "upper_alpha"),
    ]

    @pytest.mark.parametrize("text,expected", CASES)
    def test_pattern_classified(self, text, expected):
        pattern_type, _ = P.classify_pattern(text)
        assert pattern_type == expected, (
            f"'{text}' → got {pattern_type!r}, want {expected!r}"
        )

    def test_plain_text_is_none(self):
        pattern_type, _ = P.classify_pattern("이 재무제표는 한국채택국제회계기준에 따라")
        assert pattern_type == "none"


# ─────────────────────────────────────────────────────────────────────────────
# Section 11 : build_markdown  (통합 테스트)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildMarkdown:

    # ── 11-A : 넘버링 패턴 → # 헤더 승격 ─────────────────────────────────
    class TestHeadingPromotion:

        HEADING_CASES = [
            ("1. 보고기업",              "###",   "decimal_1 → ###"),
            ("2. 중요한 회계처리방침",   "###",   "decimal_1 → ###"),
            ("2.1 재무제표 작성기준",    "####",  "decimal_2 → ####"),
            ("2.1.1 세부기준",           "#####", "decimal_3 → #####"),
            ("(1) 주요 내용",            "#####", "paren_num → #####"),
            ("가. 회사가 채택한 기준서", "#####", "korean_alpha → #####"),
            ("1) 세부 항목",             "######","right-paren → ######"),
            ("① 소항목 가",             "######","circled_num → ######"),
            ("A. 기타사항",              "######","upper_alpha → ######"),
        ]

        @pytest.mark.parametrize("text,prefix,desc", HEADING_CASES)
        def test_heading_depth(self, text, prefix, desc):
            md = build_md(f"<p>{text}</p>")
            lines = [l for l in md.splitlines() if l.strip()]
            assert lines, f"출력이 비어 있음: {desc}"
            first = lines[0]
            assert first.startswith(prefix + " "), (
                f"{desc}: '{first}' 가 '{prefix} ' 로 시작해야 함"
            )

        def test_plain_sentence_not_promoted(self):
            md = build_md("<p>이 재무제표는 한국채택국제회계기준에 따라 작성되었습니다.</p>")
            assert not any(line.startswith("#") for line in md.splitlines())

        def test_heading_promotion_ignores_sentence_ending(self):
            # 넘버링 패턴이 있으면 종결어미가 있어도 헤더로 승격
            md = build_md("<p>1) 회계정책을 적용함에 있어 경영진의 판단을 요구하고 있습니다.</p>")
            lines = [l for l in md.splitlines() if l.strip()]
            assert lines[0].startswith("######"), (
                "1) 패턴이 있으므로 종결어미에도 불구하고 ###### 로 승격"
            )

        def test_continuation_stripped_from_promoted_heading(self):
            # p 태그 넘버링 헤더에 붙은 ', 계속 :' 는 헤더 승격 시 제거
            md = build_md("<p>2. 중요한 회계처리방침, 계속 :</p>")
            lines = [l for l in md.splitlines() if l.strip()]
            assert lines, "출력이 비어 있음"
            assert "계속" not in lines[0], (
                f"'계속' 꼬리가 제거되지 않음: {lines[0]!r}"
            )
            assert lines[0].startswith("### 2. 중요한 회계처리방침")

    # ── 11-B : nb 테이블 텍스트 DOM 순서 유지 ─────────────────────────────
    class TestNbTableExtraction:

        def test_nb_text_appears_in_output(self):
            md = build_md("""
                <table class="nb">
                  <tbody><tr><td>별첨 주석은 본 재무제표의 일부입니다.</td></tr></tbody>
                </table>
            """)
            assert "별첨 주석은 본 재무제표의 일부입니다." in md

        def test_nb_p_tag_extracted(self):
            md = build_md("""
                <table class="nb">
                  <tbody><tr><td>
                    <p>재 무 상 태 표</p>
                    <p>제 50 기 : 2018년 12월 31일 현재</p>
                  </td></tr></tbody>
                </table>
            """)
            assert "재 무 상 태 표" in md
            assert "제 50 기 : 2018년 12월 31일 현재" in md

        def test_nb_order_before_table(self):
            # nb제목 → TABLE참조 → nb별첨주석 순서 보장
            manifest = [P.TableExportInfo(
                table_index=1, csv_path="tables/table_0001.csv",
                row_count=2, col_count=3, unit="백만원", col_types="{}"
            )]
            md = build_md("""
                <table class="nb">
                  <tbody><tr><td>
                    <p>재 무 상 태 표</p>
                    <table class="TABLE" border="1">
                      <tr><th>과목</th><th>당기</th></tr>
                    </table>
                    <table class="nb">
                      <tbody><tr><td>별첨 주석은 본 재무제표의 일부입니다.</td></tr></tbody>
                    </table>
                  </td></tr></tbody>
                </table>
            """, manifest=manifest)

            pos_title    = md.find("재 무 상 태 표")
            pos_table    = md.find("[TABLE:")
            pos_footnote = md.find("별첨 주석")

            assert pos_title    != -1, "재무제표 제목이 출력에 없음"
            assert pos_table    != -1, "TABLE 참조가 출력에 없음"
            assert pos_footnote != -1, "별첨 주석이 출력에 없음"
            assert pos_title < pos_table < pos_footnote, (
                f"순서 오류: title={pos_title}, table={pos_table}, footnote={pos_footnote}"
            )

    # ── 11-C : 같은 tr 안 td 셀 합치기 ───────────────────────────────────
    class TestTrCellJoining:

        def test_date_parts_joined(self):
            md = build_md("""
                <table class="nb">
                  <tbody>
                    <tr>
                      <td>2018년 12월 31일</td>
                      <td>까지</td>
                    </tr>
                  </tbody>
                </table>
            """)
            assert "2018년 12월 31일 까지" in md

        def test_phone_joined(self):
            md = build_md("""
                <table class="nb">
                  <tbody>
                    <tr>
                      <td>(전   화)</td>
                      <td>031-200-1114</td>
                    </tr>
                  </tbody>
                </table>
            """)
            lines = [l for l in md.splitlines() if "031-200-1114" in l]
            assert lines, "전화번호가 출력에 없음"
            assert "전" in lines[0], "전화 레이블과 번호가 같은 줄에 있어야 함"

        def test_empty_td_skipped(self):
            md = build_md("""
                <table class="nb">
                  <tbody>
                    <tr>
                      <td>삼성전자주식회사</td>
                      <td></td>
                    </tr>
                  </tbody>
                </table>
            """)
            assert "삼성전자주식회사" in md

        def test_tr_with_block_child_not_joined(self):
            # td 안에 <p> 가 있으면 합치지 않고 각각 별도 줄 처리
            md = build_md("""
                <table class="nb">
                  <tbody>
                    <tr>
                      <td><p>재 무 상 태 표</p></td>
                      <td><p>제 50 기</p></td>
                    </tr>
                  </tbody>
                </table>
            """)
            assert "재 무 상 태 표" in md
            assert "제 50 기" in md

    # ── 11-D : TABLE 참조 인라인 삽입 ─────────────────────────────────────
    class TestTableReference:

        def test_table_ref_inserted(self):
            manifest = [P.TableExportInfo(
                table_index=1, csv_path="tables/table_0001.csv",
                row_count=2, col_count=3, unit="백만원", col_types="{}"
            )]
            md = build_md("""
                <table class="TABLE" border="1">
                  <tr><th>과목</th><th>당기</th></tr>
                  <tr><td>현금</td><td>1,000</td></tr>
                </table>
            """, manifest=manifest)
            assert "[TABLE: tables/table_0001.csv" in md
            assert "백만원" in md

        def test_table_ref_no_unit(self):
            manifest = [P.TableExportInfo(
                table_index=1, csv_path="tables/table_0001.csv",
                row_count=2, col_count=3, unit=None, col_types="{}"
            )]
            md = build_md("""
                <table class="TABLE" border="1">
                  <tr><th>과목</th><th>당기</th></tr>
                </table>
            """, manifest=manifest)
            assert "[TABLE: tables/table_0001.csv]" in md

        def test_multiple_tables_in_order(self):
            manifest = [
                P.TableExportInfo(1, "tables/t_0001.csv", 1, 1, "백만원", "{}"),
                P.TableExportInfo(2, "tables/t_0002.csv", 1, 1, "백만원", "{}"),
            ]
            md = build_md("""
                <p>재무상태표</p>
                <table class="TABLE" border="1"><tr><td>A</td></tr></table>
                <p>손익계산서</p>
                <table class="TABLE" border="1"><tr><td>B</td></tr></table>
            """, manifest=manifest)

            pos1   = md.find("t_0001.csv")
            pos2   = md.find("t_0002.csv")
            pos_bs = md.find("재무상태표")
            pos_is = md.find("손익계산서")

            assert pos_bs < pos1 < pos_is < pos2, "TABLE 참조 순서가 DOM 순서와 달라야 함"

    # ── 11-E : h 태그 → Markdown 헤더 변환 ───────────────────────────────
    class TestHTagConversion:

        @pytest.mark.parametrize("h,prefix", [
            ("h1", "#"),
            ("h2", "##"),
            ("h3", "###"),
            ("h4", "####"),
            ("h5", "#####"),
            ("h6", "######"),
        ])
        def test_h_to_markdown(self, h, prefix):
            md = build_md(f"<{h}>제목</{h}>")
            lines = [l for l in md.splitlines() if l.strip()]
            assert lines[0].startswith(prefix + " ")


# ─────────────────────────────────────────────────────────────────────────────
# Section 12 : clean_header_text / 계속 필터 (신규)
# ─────────────────────────────────────────────────────────────────────────────

class TestContinuationClean:
    """'계속' 꼬리 제거 및 독립 '계속;' 줄 필터 테스트."""

    # ── clean_header_text ──────────────────────────────────
    def test_comma_colon_removed(self):
        assert P.clean_header_text("2. 중요한 회계처리방침, 계속 :") == "2. 중요한 회계처리방침"

    def test_comma_colon_no_space_removed(self):
        assert P.clean_header_text("2.1 금융상품, 계속:") == "2.1 금융상품"

    def test_parenthesis_form_removed(self):
        assert P.clean_header_text("주석 3 (계속)") == "주석 3"

    def test_trailing_comma_cleaned(self):
        # 계속 제거 후 남은 후행 콤마도 제거
        assert P.clean_header_text("가. 금융자산, 계속") == "가. 금융자산"

    def test_normal_title_unchanged(self):
        assert P.clean_header_text("2. 중요한 회계처리방침") == "2. 중요한 회계처리방침"

    def test_pure_continuation_becomes_empty(self):
        assert P.clean_header_text("계속") == ""

    # ── 독립 '계속;' 줄 필터 (build_markdown 최종 단계) ───
    def test_standalone_gyesok_filtered_p(self):
        # <p>계속;</p> 단독 → 출력에서 제거
        md = build_md("<p>계속;</p>")
        assert md.strip() == ""

    def test_standalone_gyesok_no_semicolon(self):
        # 세미콜론 없는 단독 "계속" 도 제거
        md = build_md("<p>계속</p>")
        assert md.strip() == ""

    def test_gyesok_in_sentence_preserved(self):
        # 문장 중간의 "계속" 은 제거되면 안 됨
        md = build_md("<p>사업을 계속 영위하기 위해 노력합니다.</p>")
        assert "계속" in md

    def test_context_before_after_preserved(self):
        # [TABLE:...] → 계속; → ### 헤더 순서에서 TABLE과 헤더는 보존
        manifest = [P.TableExportInfo(1, "tables/t_0001.csv", 1, 1, None, "{}")]
        md = build_md("""
            <table class="TABLE" border="1"><tr><td>A</td></tr></table>
            <p>계속;</p>
            <p>2. 중요한 회계처리방침</p>
        """, manifest=manifest)
        assert "[TABLE:" in md
        assert "2. 중요한 회계처리방침" in md
        assert "계속" not in md