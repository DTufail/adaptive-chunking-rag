"""
squad_v2_loader.py
───────────────────
Explicit SQuAD v2 dataset loader.
"""

import json
import os
from datasets import load_dataset


def load_squad_v2(split="validation"):
    """
    Load SQuAD 2.0 dataset split.
    Returns a dict:
      'data': list of dicts with keys: 'context', 'question', 'answer', 'answer_start'
      'summary': count of answerable and unanswerable questions
    """
    dataset = load_dataset("rajpurkar/squad_v2", split=split)
    data_list = []
    answerable_count = 0
    unanswerable_count = 0

    for item in dataset:
        answers = item['answers']
        if answers['text']:
            answer = answers['text'][0]
            answer_start = answers['answer_start'][0]
            answerable_count += 1
        else:
            answer = ""
            answer_start = -1
            unanswerable_count += 1

        data_list.append({
            "context": item["context"],
            "question": item["question"],
            "answer": answer,
            "answer_start": answer_start
        })

    summary = {
        "total": len(dataset),
        "answerable": answerable_count,
        "unanswerable": unanswerable_count
    }

    return {"data": data_list, "summary": summary}


def save_subset(data_dict, n, filename):
    """
    Save the first n examples and summary to a JSON file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    subset = data_dict["data"][:n]
    output = {
        "summary": data_dict["summary"],
        "examples": subset
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(subset)} examples to {filename}")
    print("Summary:", data_dict["summary"])


if __name__ == "__main__":
    for split in ["train", "validation"]:
        data_dict = load_squad_v2(split=split)
        save_subset(data_dict, 100, f"data/{split}_100.json")
        save_subset(data_dict, 1000, f"data/{split}_1000.json")
