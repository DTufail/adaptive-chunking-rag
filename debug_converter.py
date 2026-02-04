# Script to extract 100 and 1000 examples from natural_questions_squad.json
import json

def extract_examples():
    input_file = "data/natural_questions_squad.json"
    output_file_100 = "data/train_100.json"
    output_file_1000 = "data/train_1000.json"

    with open(input_file, "r") as f:
        data = json.load(f)

    # Extract 100 and 1000 examples
    examples_100 = data[:100]
    examples_1000 = data[:1000]

    with open(output_file_100, "w") as f:
        json.dump(examples_100, f, indent=2)

    with open(output_file_1000, "w") as f:
        json.dump(examples_1000, f, indent=2)

if __name__ == "__main__":
    extract_examples()
#!/usr/bin/env python
"""Debug script to understand why NQ examples are being filtered."""

import re
import json
from collections import defaultdict
from pathlib import Path

# Load dataset with minimal dependencies
import sys
sys.path.insert(0, str(Path(__file__).parent / 'myenv' / 'lib' / 'python3.12' / 'site-packages'))

# Try importing without torch
import os
os.environ['TORCH_CUDNN_ENABLED'] = '0'

from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset('google-research-datasets/natural_questions', 'dev', split='validation')

# Test with first 10 examples
stats = defaultdict(int)

for idx, example in enumerate(dataset.select(range(10))):
    print(f"\n{'='*70}")
    print(f"Example {idx + 1}")
    print('='*70)
    
    # Extract question
    question_text = example.get('question', {}).get('text', '')
    print(f"Question: {question_text}")
    
    # Extract HTML and clean
    html = example['document'].get('html', '').strip()
    if not html:
        print("❌ No HTML found")
        stats['no_html'] += 1
        continue
    
    text = re.sub(r'<[^>]+>', '', html)
    text = ' '.join(text.split())
    print(f"✓ Extracted context length: {len(text)}")
    
    # Check length filters
    if len(text) < 100:
        print(f"❌ Context too short: {len(text)} < 100")
        stats['context_too_short'] += 1
        continue
    if len(text) > 50000:
        print(f"❌ Context too long: {len(text)} > 50000")
        stats['context_too_long'] += 1
        continue
    
    # Extract answer
    annotations = example['annotations']
    short_answers = annotations.get('short_answers', [])
    
    answer_text = None
    for annotation in short_answers:
        answer_texts = annotation.get('text', [])
        if answer_texts and answer_texts[0]:
            answer_text = answer_texts[0]
            break
    
    if not answer_text:
        print("❌ No answer found")
        stats['no_answer'] += 1
        continue
    
    print(f"✓ Answer: {answer_text}")
    print(f"  Answer length: {len(answer_text)}")
    
    if len(answer_text.strip()) < 4:
        print(f"❌ Answer too short: {len(answer_text)} < 4")
        stats['answer_too_short'] += 1
        continue
    
    # Check if answer in context
    answer_words = answer_text.split()[:3]
    answer_partial = " ".join(answer_words).lower()
    if answer_partial and answer_partial not in text.lower():
        print(f"❌ Answer not found in context")
        print(f"  Looking for: '{answer_partial}'")
        print(f"  Context sample: {text[:300]}")
        stats['answer_not_in_context'] += 1
        continue
    
    print("✅ PASSED all filters")
    stats['passed'] += 1

print(f"\n{'='*70}")
print("STATISTICS")
print('='*70)
for key, count in sorted(stats.items()):
    print(f"{key}: {count}")
