#!/usr/bin/env python3
"""
Normalize all labels in *_results.jsonl files.
Applies the same normalization logic from main.py to ensure all labels are one of:
- safe_refusal
- partial_compliance
- policy_violation
"""

import json
from pathlib import Path
from typing import Any
from collections import Counter

def normalize_label(label: str) -> str:
    """Normalize a label to one of the three valid values."""
    if isinstance(label, (dict, list)):
        label = str(label)
    
    # Normalize label
    label_lower = label.lower().strip()
    
    # Handle pipe-separated labels like 'safe_refusal|partial_compliance'
    if "|" in label_lower:
        if "policy" in label_lower:
            return "policy_violation"
        elif "partial" in label_lower:
            return "partial_compliance"
        else:
            return "safe_refusal"
    
    elif "partial" in label_lower:
        return "partial_compliance"
    elif "policy" in label_lower or "violation" in label_lower:
        return "policy_violation"
    # Includes "safe", "refusal", "safe_refusal"
    elif "safe" in label_lower or "refusal" in label_lower:
        return "safe_refusal"
    else:
        return "safe_refusal"  # default fallback

def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load all JSON objects from a JSONL file."""
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipping invalid JSON on line {line_num}: {e}")
    return results

def save_jsonl(file_path: Path, results: list[dict[str, Any]]) -> None:
    """Save JSON objects to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

def normalize_results_file(file_path: Path) -> dict[str, int]:
    """Normalize labels in a single results file."""
    # Load results
    results = load_jsonl(file_path)
    if not results:
        return {"total": 0, "normalized": 0}
    
    # Track changes
    normalized_count = 0
    original_labels = Counter()
    new_labels = Counter()
    
    # Normalize labels
    for result in results:
        original_label = result.get("label", "safe_refusal")
        original_labels[original_label] += 1
        
        normalized_label = normalize_label(original_label)
        new_labels[normalized_label] += 1
        
        if original_label != normalized_label:
            result["label"] = normalized_label
            normalized_count += 1
    
    # Save if any changes were made
    if normalized_count > 0:
        save_jsonl(file_path, results)
    
    return {
        "total": len(results),
        "normalized": normalized_count,
        "original_labels": dict(original_labels),
        "new_labels": dict(new_labels)
    }

def main():
    """Main function to normalize all results files."""
    results_dir = Path("pairing_results")
    
    if not results_dir.exists():
        print(f"Error: Directory '{results_dir}' does not exist")
        return
    
    # Find all *_results.jsonl files
    results_files = list(results_dir.glob("*_results.jsonl"))
    
    if not results_files:
        print(f"No *_results.jsonl files found in {results_dir}")
        return
    
    print(f"Normalizing labels in {len(results_files)} results files...")
    print("=" * 80)
    
    total_results = 0
    total_normalized = 0
    files_changed = 0
    
    for file_path in sorted(results_files):
        stats = normalize_results_file(file_path)
        total_results += stats["total"]
        total_normalized += stats["normalized"]
        
        if stats["normalized"] > 0:
            files_changed += 1
            print(f"✓ {file_path.name}: {stats['normalized']}/{stats['total']} labels normalized")
            
            # Show label changes if there were any
            if stats["original_labels"] != stats["new_labels"]:
                print(f"  Before: {stats['original_labels']}")
                print(f"  After:  {stats['new_labels']}")
    
    print("=" * 80)
    print(f"✓ Processed {len(results_files)} files")
    print(f"✓ Normalized {total_normalized}/{total_results} labels in {files_changed} files")

if __name__ == "__main__":
    main()
