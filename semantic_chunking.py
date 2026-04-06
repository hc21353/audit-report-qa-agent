import os
import re
import json
import pandas as pd
from pathlib import Path

def normalize_korean(text, join_chars=False):
    """
    Normalizes Korean text by addressing patternless spaces and artifacts.
    Tier 1: Join known financial terms.
    Tier 2: Collapse multiple whitespaces and Unicode artifacts.
    Tier 3: Join single characters separated by spaces (for original field only).
    """
    if not isinstance(text, str) or not text:
        return text
    
    # Tier 2: Normalize all whitespace
    text = re.sub(r'[\s\xa0]+', ' ', text).strip()
    
    # Tier 1: Known Financial Terms (Must-Join)
    financial_terms = [
        "재무제표", "현금흐름표", "재무상태표", "포괄손익계산서", "자본변동표", 
        "이익잉여금", "백만원", "감사보고서", "삼성전자", "현금및현금성자산", 
        "단기금융상품", "매출채권", "재고자산", "유형자산", "무형자산", 
        "영업활동", "투자활동", "재무활동", "있습니다", "합니다", "부채", "자산", "자본"
    ]
    for term in financial_terms:
        pattern = r'\s*'.join([re.escape(c) for c in term])
        text = re.sub(pattern, term, text)
    
    # Tier 3: Aggressive Character-Level Joining (for Search Headings)
    if join_chars:
        text = re.sub(r'(([가-힣])\s){2,}([가-힣])', lambda m: m.group(0).replace(" ", ""), text)
        
    return text.strip()

def get_row_original_text(row, col_names, unit, table_anchor, global_context):
    """
    Creates a 'Golden Standard' search string for table rows.
    Format: [Global Context][Table Anchor] Column: Value | ...
    """
    parts = []
    for i, col in enumerate(col_names):
        val = str(row.iloc[i]).strip()
        if not val or val == 'nan' or val == '-' or val == '0.0':
            continue
        clean_col = normalize_korean(col, join_chars=True)
        clean_val = normalize_korean(val, join_chars=False)
        parts.append(f"{clean_col}: {clean_val}")
    
    row_str = " | ".join(parts) + f" (단위: {normalize_korean(unit)})"
    
    # Anchor the table name into the search string
    anchor_prefix = f"[{normalize_korean(table_anchor, join_chars=True)}]" if table_anchor else ""
    
    final_text = f"{global_context} {anchor_prefix} {row_str}"
    return normalize_korean(final_text, join_chars=True)

def is_noise(text):
    """Filters out noise to keep the Vector DB clean."""
    clean_text = text.replace(" ", "")
    noise_patterns = [
        r'^삼성전자주식회사$',
        r'^제\d+기$',
        r'^\d{4}년\d{2}월\d{2}일부터$',
        r'^\d{4}년\d{2}월\d{2}일까지$',
        r'^"첨부된재무제표는당사가작성한것입니다\."'
    ]
    for pattern in noise_patterns:
        if re.match(pattern, clean_text):
            return True
    return len(clean_text) < 4

def process_year(output_dir):
    year_match = re.search(r'20\d{2}', output_dir.name)
    if not year_match: return []
    year = int(year_match.group())
    
    md_file = output_dir / f"{output_dir.name}.md"
    if not md_file.exists(): return []

    with open(md_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chunks = []
    current_section_path = ["Root"]
    period = f"제 {year - 1968} 기"
    
    doc_type = "감사보고서"
    for line in lines[:50]:
        if "연결" in line: doc_type = "연결 감사보고서"
        m = re.search(r'제\s*(\d+)\s*기', line)
        if m: period = normalize_korean(m.group(0))

    global_context = f"[삼성전자 {year}년 {period} {doc_type}]"
    doc_id = f"SAM_{year}_ANNUAL"
    
    buffer = []
    pseudo_heading = None

    def flush_narrative():
        nonlocal pseudo_heading, buffer
        if not buffer: return
            
        content_text = " ".join(buffer)
        clean_content = normalize_korean(content_text, join_chars=False)
        clean_pseudo = normalize_korean(pseudo_heading, join_chars=False) if pseudo_heading else None
        
        original_text = f"{global_context} " + (f"[{clean_pseudo}] {clean_content}" if clean_pseudo else clean_content)
        original_text = normalize_korean(original_text, join_chars=True)
            
        if not is_noise(original_text):
            path_str = " > ".join([normalize_korean(p, join_chars=True) for p in current_section_path])
            chunk_type = "Note" if "주석" in path_str else "Narrative"
            
            chunks.append({
                "text": {
                    "original": original_text,
                    "struct": {"pseudo_heading": clean_pseudo, "content": clean_content}
                },
                "metadata": {
                    "doc_id": doc_id, "fiscal_year": year, "period": period,
                    "section_path": path_str, "chunk_type": chunk_type,
                    "is_consolidated": "연결" in path_str
                }
            })
        buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            flush_narrative()
            continue
        
        head_match = re.match(r'^(#+)\s+(.*)', line)
        if head_match:
            flush_narrative()
            pseudo_heading = None
            level = len(head_match.group(1))
            title = head_match.group(2)
            if level <= len(current_section_path):
                current_section_path = current_section_path[:level-1]
            current_section_path.append(title)
            continue

        table_match = re.match(r'\[TABLE:\s*(tables/table_\d+\.csv)\s*단위:\s*([^\]]+)\]', line)
        if table_match:
            flush_narrative()
            csv_path = table_match.group(1)
            unit = table_match.group(2)
            table_anchor = pseudo_heading
            
            full_csv_path = output_dir / csv_path
            if full_csv_path.exists():
                try:
                    df = pd.read_csv(full_csv_path)
                    col_names = df.columns.tolist()
                    path_str = " > ".join([normalize_korean(p, join_chars=True) for p in current_section_path])
                    for _, row in df.iterrows():
                        orig_text = get_row_original_text(row, col_names, unit, table_anchor, global_context)
                        if len(orig_text) > 30:
                            notes = []
                            if len(col_names) > 1 and ('주석' in col_names[1] or '주 석' in col_names[1]):
                                note_val = str(row.iloc[1]).strip()
                                if note_val and note_val != 'nan' and note_val != '-':
                                    notes = [f"주석 {n.strip()}" for n in re.split(r'[,&]', note_val) if n.strip()]

                            clean_row = {normalize_korean(k): normalize_korean(str(v)) for k, v in row.to_dict().items()}
                            chunks.append({
                                "text": {
                                    "original": orig_text,
                                    "struct": {
                                        "row_data": clean_row, "unit": normalize_korean(unit),
                                        "table_ref": csv_path, "table_anchor": table_anchor
                                    }
                                },
                                "metadata": {
                                    "doc_id": doc_id, "fiscal_year": year, "period": period,
                                    "section_path": path_str, "chunk_type": "Table_Row",
                                    "note_refs": notes, "is_consolidated": "연결" in path_str
                                }
                            })
                except Exception as e:
                    print(f"Error processing table {csv_path}: {e}")
            pseudo_heading = None
            continue

        if not buffer and len(line) < 40 and not line.endswith(('.', ':', ';')):
            pseudo_heading = line
        else:
            buffer.append(line)
            
    flush_narrative()
    return chunks

def main():
    base_dir = Path(__file__).parent / "parsed_data" / "output"
    all_chunks = []
    
    for year_dir in sorted(base_dir.iterdir()):
        if year_dir.is_dir() and year_dir.name.startswith("감사보고서_"):
            print(f"Processing {year_dir.name}...")
            year_chunks = process_year(year_dir)
            all_chunks.extend(year_chunks)
            
    output_file = Path(__file__).parent / "parsed_data" / "chunks" / "semantic_chunks_tagged.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"Finished. Total chunks: {len(all_chunks)}")

if __name__ == "__main__":
    main()
