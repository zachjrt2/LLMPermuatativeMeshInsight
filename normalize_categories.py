#!/usr/bin/env python3
"""
Normalize attack_category labels in *_results.jsonl files.
Adapted from normalize_categories_v2.py in the root qwen directory.
Maps messy LLM-generated category strings to canonical categories:
  - Harmful Instructions
  - Inappropriate Content
  - Jailbreak
  - Sensitive Information
  - Physical Harm / Violence
  - Hacking / Malware
  - Fraud / Deception
  - Sexual Content
  - Hate Speech
  - Other
"""

import json
import os
import shutil
from pathlib import Path
from collections import Counter


def normalize_category(category) -> str:
    """Normalize an attack category to one of the canonical values."""
    if not category:
        return "Other"

    # Handle list-formatted categories (e.g. from LLMs returning JSON arrays)
    if isinstance(category, list):
        category = ", ".join(str(c) for c in category)
    category = str(category)
    cat_lower = category.lower().strip()

    # Skip clearly garbage / placeholder values
    garbage = [
        "your attack category here",
        "your category name here",
        "not-part-of-category",
        "ceo", "bornin", "coherence",
        "ability to classify",
        "syntax errors", "invalid_json", "invalid_api",
        "exceeds_threshold", "humorous", "innovation",
        "poem", "winter", "flower", "llm",
        "harmless_program", "harmless",
        "real life scenario",
        "one of: poem",
        "attack_prompt_set",
        "crash",
        "generation of highly realistic and convincing text",
        "none", "unknown", "unintentional errors",
    ]
    for g in garbage:
        if g in cat_lower:
            return "Other"

    # ── Explicit exact matches (fast path) ──────────────────────────
    if cat_lower in ("harmful_instructions", "harmful instructions"):
        return "Harmful Instructions"
    if cat_lower in ("inappropriate_content", "inappropriate content", "inappropriate_code"):
        return "Inappropriate Content"
    if cat_lower in ("jailbreak",):
        return "Jailbreak"
    if cat_lower in ("sensitive_info", "sensitive information"):
        return "Sensitive Information"

    # ── Broad keyword matching (order matters) ──────────────────────

    # Sexual content
    if any(k in cat_lower for k in ("sex", "porn", "erotic", "adult", "nsfw", "bestiality")):
        return "Sexual Content"

    # Hate speech
    if any(k in cat_lower for k in ("hate", "discriminat", "racis", "sexi", "bias", "propaganda", "exclusion")):
        return "Hate Speech"

    # Physical harm / violence
    if any(k in cat_lower for k in (
        "violence", "physical_harm", "weapon", "bomb", "kill",
        "suicide", "terror", "nuclear", "firework", "dangerous",
        "eating disorder", "drinking", "attack plan", "attack_plan",
    )):
        return "Physical Harm / Violence"

    # Fraud / deception
    if any(k in cat_lower for k in (
        "fraud", "deception", "social_engineer", "social engineer",
        "phishing", "scam", "fake", "deepfake", "misinformation",
        "wrong_information", "wrong information", "inaccurate",
    )):
        return "Fraud / Deception"

    # Hacking / malware
    if any(k in cat_lower for k in (
        "hack", "malware", "malicious", "exploit", "virus", "trojan",
        "worm", "injection", "cyber", "crack", "unauthorized",
        "security_breach", "security-bypass", "security_information",
        "system_hacking", "website_hacking", "website modification",
        "inline_code", "infectious", "infect", "jeopardize", "zero",
        "internet_security", "inadequate_security", "internet security", 
        "inadequate security",
    )):
        return "Hacking / Malware"

    # Jailbreak / bypass
    if any(k in cat_lower for k in (
        "jailbreak", "bypass", "override", "restriction",
        "adversarial", "safety_guidelines", "safety guidelines",
    )):
        return "Jailbreak"

    # Sensitive information / privacy
    if any(k in cat_lower for k in (
        "sensitive", "privacy", "private", "credential", "password",
        "pii", "reveal", "confidential", "evidence_planting", "exposure",
    )):
        return "Sensitive Information"

    # Harmful instructions (must come after specific checks containing "harm")
    if any(k in cat_lower for k in (
        "harmful", "unsafe", "illegal", "unlawful",
        "hypothetical_harm", "invasive",
    )):
        return "Harmful Instructions"

    # Inappropriate content
    if any(k in cat_lower for k in ("inappropriate", "offensive", "unethical", "improper")):
        return "Inappropriate Content"

    # Catch-all for "one of: ..." patterns
    if "one of" in cat_lower:
        if "harmful" in cat_lower:
            return "Harmful Instructions"
        if "inappropriate" in cat_lower:
            return "Inappropriate Content"
        if "jailbreak" in cat_lower:
            return "Jailbreak"
        if "sensitive" in cat_lower:
            return "Sensitive Information"
        if "danger" in cat_lower:
            return "Physical Harm / Violence"
        if "adversarial" in cat_lower or "safety" in cat_lower:
            return "Jailbreak"
        if "malware" in cat_lower or "hack" in cat_lower or "exploit" in cat_lower:
            return "Hacking / Malware"

    return "Other"


def process_files(results_dir: Path, dry_run: bool = False):
    """Normalize attack_category in all *_results.jsonl and *_prompts.jsonl files."""
    files = sorted(list(results_dir.glob("*_results.jsonl")) + list(results_dir.glob("*_prompts.jsonl")))
    print(f"Found {len(files)} files in {results_dir}")

    before_counts: Counter = Counter()
    after_counts: Counter = Counter()
    total_changed = 0
    files_changed = 0

    for file_path in files:
        temp_path = file_path.with_suffix(".jsonl.tmp")
        modified = False

        try:
            with open(file_path, "r", encoding="utf-8") as f_in, \
                 open(temp_path, "w", encoding="utf-8") as f_out:

                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        original_cat = data.get("attack_category", "")
                        new_cat = normalize_category(original_cat)

                        before_counts[str(original_cat)] += 1
                        after_counts[new_cat] += 1

                        if str(original_cat) != new_cat:
                            data["attack_category"] = new_cat
                            modified = True
                            total_changed += 1

                        f_out.write(json.dumps(data) + "\n")
                    except json.JSONDecodeError:
                        print(f"  Skipping invalid JSON in {file_path.name}")

            if modified and not dry_run:
                files_changed += 1
                backup_path = file_path.with_suffix(".jsonl.catbak")
                if not backup_path.exists():
                    shutil.copy2(file_path, backup_path)
                shutil.move(str(temp_path), str(file_path))
                print(f"  Updated {file_path.name}")
            else:
                if temp_path.exists():
                    os.remove(temp_path)

        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            if temp_path.exists():
                os.remove(temp_path)

    # Summary
    print("\n" + "=" * 60)
    print(f"Total results processed: {sum(before_counts.values())}")
    print(f"Categories changed: {total_changed}")
    print(f"Files modified: {files_changed}")
    if dry_run:
        print("(DRY RUN — no files were modified)")

    print("\nNormalized category distribution:")
    print("-" * 40)
    for cat, count in after_counts.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize attack_category labels in results files.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    parser.add_argument("--dir", type=str, default="pairing_results", help="Results directory (default: pairing_results)")
    args = parser.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.exists():
        print(f"Error: '{results_dir}' does not exist")
        exit(1)

    process_files(results_dir, dry_run=args.dry_run)
