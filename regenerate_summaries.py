#!/usr/bin/env python3
"""
Regenerate all summary files from existing result files
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, UTC
from pathlib import Path

# All models
ALL_MODELS = [
    "llama3.1:8b",
    "mistral",
    "neural-chat",
    "orca-mini",
    "openhermes",
    "nous-hermes2",
    "phi3:mini",
    "falcon",
    "gemma2",
    "granite4",
]

results_dir = Path("pairing_results")

def load_results(results_file):
    """Load all results from JSONL file."""
    if not results_file.exists():
        return []
    
    results = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results

def summarize_results(results, pairing_id, red_team_model, judge_model):
    """Generate summary statistics from results."""
    if not results:
        return {}
    
    overall_labels = Counter()
    overall_danger_scores = []
    model_stats = defaultdict(lambda: {
        "total": 0,
        "safe_refusal": 0,
        "partial_compliance": 0,
        "policy_violation": 0,
        "danger_scores": []
    })
    
    for result in results:
        label = result.get("label", "safe_refusal")
        danger_score = result.get("danger_score", 0)
        target_model = result.get("target_model", "unknown")
        
        overall_labels[label] += 1
        overall_danger_scores.append(danger_score)
        
        model_stats[target_model]["total"] += 1
        # Increment the specific label count
        if label in model_stats[target_model]:
            model_stats[target_model][label] += 1
        else:
            model_stats[target_model][label] = 1
        model_stats[target_model]["danger_scores"].append(danger_score)
    
    for model, stats in model_stats.items():
        total = stats["total"]
        # Use .get() to safely access label counts
        safe_refusal = stats.get("safe_refusal", 0)
        partial_compliance = stats.get("partial_compliance", 0)
        policy_violation = stats.get("policy_violation", 0)
        
        stats["safe_refusal_pct"] = (safe_refusal / total * 100) if total > 0 else 0
        stats["partial_compliance_pct"] = (partial_compliance / total * 100) if total > 0 else 0
        stats["policy_violation_pct"] = (policy_violation / total * 100) if total > 0 else 0
        
        scores = stats["danger_scores"]
        stats["avg_danger_score"] = sum(scores) / len(scores) if scores else 0
        stats["max_danger_score"] = max(scores) if scores else 0
        stats["min_danger_score"] = min(scores) if scores else 0
        del stats["danger_scores"]
    
    summary = {
        "pairing_id": pairing_id,
        "red_team_model": red_team_model,
        "judge_model": judge_model,
        "total_rounds": len(results),
        "total_models": len(model_stats),
        "overall_labels": dict(overall_labels),
        "overall_avg_danger_score": sum(overall_danger_scores) / len(overall_danger_scores) if overall_danger_scores else 0,
        "model_stats": dict(model_stats),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    
    return summary

print("Regenerating all summary files...")
print("=" * 80)

# Generate all pairings
all_pairings = []
for red_team in ALL_MODELS:
    for judge in ALL_MODELS:
        if red_team == judge:
            continue
        pairing_id = f"redteam-{red_team.replace(':', '-')}_judge-{judge.replace(':', '-')}"
        all_pairings.append((pairing_id, red_team, judge))

regenerated_count = 0
for pairing_id, red_team, judge in all_pairings:
    results_file = results_dir / f"{pairing_id}_results.jsonl"
    summary_file = results_dir / f"{pairing_id}_summary.json"
    
    if not results_file.exists():
        print(f"✗ {pairing_id}: Results file missing, skipping")
        continue
    
    # Load results and generate summary
    results = load_results(results_file)
    if not results:
        print(f"✗ {pairing_id}: No results found in file, skipping")
        continue
    
    summary = summarize_results(results, pairing_id, red_team, judge)
    
    # Save summary
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    regenerated_count += 1

print("=" * 80)
print(f"Regenerated {regenerated_count} summary files")
print()

# Now regenerate master summary
print("Regenerating master_summary.json...")
print("=" * 80)

master_summary = {
    "total_pairings": len(all_pairings),
    "models_tested": ALL_MODELS,
    "attack_rounds_per_pairing": 30,
    "max_prompt_retries": 3,
    "max_test_retries": 3,
    "json_translator_model": "llama3.1:8b",
    "pairings": [],
    "generated_at": datetime.now(UTC).isoformat(),
}

found_count = 0
for pairing_id, red_team, judge in all_pairings:
    summary_file = results_dir / f"{pairing_id}_summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
            master_summary["pairings"].append({
                "pairing_id": pairing_id,
                "red_team_model": red_team,
                "judge_model": judge,
                "overall_avg_danger_score": summary.get("overall_avg_danger_score", 0),
                "overall_labels": summary.get("overall_labels", {}),
            })
        found_count += 1

master_summary_file = results_dir / "master_summary.json"
with open(master_summary_file, "w", encoding="utf-8") as f:
    json.dump(master_summary, f, indent=2)

print(f"✓ Master summary updated with {found_count}/{len(all_pairings)} pairings")
print(f"✓ Saved to {master_summary_file}")
print("=" * 80)
