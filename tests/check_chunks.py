import json
from collections import Counter
from pathlib import Path

# 파일 경로를 입력하세요
file_path = Path(__file__).parent.parent / "parsed_data" / "chunks" / "semantic_chunks_tagged.jsonl"

total_chunks = 0
year_counts = Counter()
chunk_type_counts = Counter()

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
            
        try:
            data = json.loads(line)
            total_chunks += 1
            
            # metadata 추출
            metadata = data.get("metadata", {})
            
            # 연도별 개수 카운트
            year = metadata.get("fiscal_year", "Unknown")
            year_counts[year] += 1
            
            # chunk_type 분포 카운트
            chunk_type = metadata.get("chunk_type", "Unknown")
            chunk_type_counts[chunk_type] += 1
            
        except json.JSONDecodeError:
            pass

# 결과 출력
print(f"■ 총 청크 수 (Total Chunks): {total_chunks:,}개\n")

print("■ 연도별 청크 개수:")
for year, count in sorted(year_counts.items()):
    print(f"  - {year}년: {count:,}개")

print("\n■ Chunk Type 분포:")
for ctype, count in sorted(chunk_type_counts.items()):
    print(f"  - {ctype}: {count:,}개")