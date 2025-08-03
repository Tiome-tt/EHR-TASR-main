import json
from pathlib import Path
import sys
from config.config import ROOT, DATASET, TASK
import os
os.chdir(ROOT)

FILE_PATH = Path(f"data/{DATASET}/preprocessed/reasoning_outputs/{TASK.lower()}_reasoning_chain.jsonl")

def should_drop(text: str) -> bool:
    if text is None or text.strip() == "":
        return True
    lower = text.lower()
    return ("is 1" in lower) or ("is 0" in lower)

def main() -> None:
    if not FILE_PATH.exists():
        sys.exit(f"can't find file：{FILE_PATH}")

    kept_records = []
    dropped = 0

    with FILE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if should_drop(obj.get("Reasoning_chain", "")):
                dropped += 1
            else:
                kept_records.append(obj)

    # 覆盖写回
    with FILE_PATH.open("w", encoding="utf-8") as f:
        for obj in kept_records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Overwrite saved: kept {len(kept_records)} records, deleted {dropped}.")

if __name__ == "__main__":
    main()
