def write_jsonl(rows, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.json())
            f.write("\n")
