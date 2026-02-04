import json
import random
from pathlib import Path

INPUT_PATH = Path("/Users/daniyal/adaptive-chunking-rag/data/natural_questions_squad.json")
OUT_100 = INPUT_PATH.with_name("natural_questions_squad_100.json")
OUT_1000 = INPUT_PATH.with_name("natural_questions_squad_1000.json")

RANDOM_SEED = 42

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    root = json.load(f)

if "examples" not in root or not isinstance(root["examples"], list):
    raise ValueError("Expected top-level key 'examples' containing a list")

data = root["examples"]

if len(data) < 1000:
    raise ValueError(f"Dataset too small: {len(data)} examples")

random.seed(RANDOM_SEED)
shuffled = random.sample(data, len(data))

sample_100 = shuffled[:100]
sample_1000 = shuffled[:1000]

out_100 = {"examples": sample_100}
out_1000 = {"examples": sample_1000}

with open(OUT_100, "w", encoding="utf-8") as f:
    json.dump(out_100, f, ensure_ascii=False, indent=2)

with open(OUT_1000, "w", encoding="utf-8") as f:
    json.dump(out_1000, f, ensure_ascii=False, indent=2)

print("Wrote:")
print(f"- {OUT_100} ({len(sample_100)} examples)")
print(f"- {OUT_1000} ({len(sample_1000)} examples)")
