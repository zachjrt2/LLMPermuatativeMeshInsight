#!/usr/bin/env python3
"""Quick verification script to check if a results file is sorted correctly."""

import json
from pathlib import Path

# Check a sample file
file_path = Path("pairing_results/redteam-falcon_judge-granite4_results.jsonl")

with open(file_path, "r", encoding="utf-8") as f:
    results = [json.loads(line) for line in f if line.strip()]

print(f"Checking: {file_path.name}")
print(f"Total entries: {len(results)}")
print()

print("First 5 entries:")
for r in results[:5]:
    print(f"  Round {r['round']}, Model: {r['target_model']}")

print()
print("Last 5 entries:")
for r in results[-5:]:
    print(f"  Round {r['round']}, Model: {r['target_model']}")

print()

# Verify sorting
is_sorted = True
for i in range(len(results) - 1):
    curr = results[i]
    next_r = results[i + 1]
    
    # Check if sorted by round first, then by target_model
    if curr['round'] > next_r['round']:
        is_sorted = False
        print(f"❌ Sorting error at index {i}: round {curr['round']} > {next_r['round']}")
        break
    elif curr['round'] == next_r['round'] and curr['target_model'] > next_r['target_model']:
        is_sorted = False
        print(f"❌ Sorting error at index {i}: same round but {curr['target_model']} > {next_r['target_model']}")
        break

if is_sorted:
    print("✓ File is correctly sorted by round, then by target_model")
