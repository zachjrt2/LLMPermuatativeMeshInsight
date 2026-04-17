#!/usr/bin/env python3
"""
Organize all *_results.jsonl files in pairing_results/ directory.
Sorts each file first by round (ascending), then by target_model (ascending).
"""

import json
from pathlib import Path
from typing import Any

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

def organize_results_file(file_path: Path) -> None:
    """Organize a single results file by round and target_model."""
    print(f"Processing: {file_path.name}")
    
    # Load results
    results = load_jsonl(file_path)
    if not results:
        print(f"  No valid results found, skipping")
        return
    
    original_count = len(results)
    
    # Sort by round (int), then by target_model (str)
    results.sort(key=lambda x: (x.get("round", 0), x.get("target_model", "")))
    
    # Save sorted results
    save_jsonl(file_path, results)
    
    print(f"  ✓ Sorted {original_count} results")

def main():
    """Main function to organize all results files."""
    results_dir = Path("pairing_results")
    
    if not results_dir.exists():
        print(f"Error: Directory '{results_dir}' does not exist")
        return
    
    # Find all *_results.jsonl files
    results_files = list(results_dir.glob("*_results.jsonl"))
    
    if not results_files:
        print(f"No *_results.jsonl files found in {results_dir}")
        return
    
    print(f"Found {len(results_files)} results files to organize")
    print("=" * 80)
    
    for file_path in sorted(results_files):
        organize_results_file(file_path)
    
    print("=" * 80)
    print(f"✓ Organized {len(results_files)} files")

if __name__ == "__main__":
    main()
