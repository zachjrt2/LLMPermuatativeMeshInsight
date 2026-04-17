#!/usr/bin/env python3
"""
Generate enhanced master_summary.json with target model performance groupings.
Shows how each target model performed against different red teams and judges.
"""

import json
from collections import defaultdict, Counter
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

print("Generating enhanced master summary...")
print("=" * 80)

# Generate all pairings
all_pairings = []
for red_team in ALL_MODELS:
    for judge in ALL_MODELS:
        if red_team == judge:
            continue
        pairing_id = f"redteam-{red_team.replace(':', '-')}_judge-{judge.replace(':', '-')}"
        all_pairings.append((pairing_id, red_team, judge))

# Collect all results for aggregation
all_results = []
pairing_summaries = []

for pairing_id, red_team, judge in all_pairings:
    results_file = results_dir / f"{pairing_id}_results.jsonl"
    summary_file = results_dir / f"{pairing_id}_summary.json"
    
    if not results_file.exists():
        continue
    
    # Load results
    results = load_results(results_file)
    for result in results:
        result['red_team_model'] = red_team
        result['judge_model'] = judge
        all_results.append(result)
    
    # Load summary
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
            pairing_summaries.append({
                "pairing_id": pairing_id,
                "red_team_model": red_team,
                "judge_model": judge,
                "overall_avg_danger_score": summary.get("overall_avg_danger_score", 0),
                "overall_labels": summary.get("overall_labels", {}),
            })

print(f"Loaded {len(all_results)} total results from {len(pairing_summaries)} pairings")

# Aggregate by target model vs red team
target_vs_redteam = defaultdict(lambda: defaultdict(lambda: {
    "total_tests": 0,
    "labels": Counter(),
    "danger_scores": []
}))

for result in all_results:
    target = result.get("target_model")
    red_team = result.get("red_team_model")
    label = result.get("label", "safe_refusal")
    danger_score = result.get("danger_score", 0)
    
    target_vs_redteam[target][red_team]["total_tests"] += 1
    target_vs_redteam[target][red_team]["labels"][label] += 1
    target_vs_redteam[target][red_team]["danger_scores"].append(danger_score)

# Convert to final format
target_vs_redteam_final = {}
for target, red_teams in target_vs_redteam.items():
    target_vs_redteam_final[target] = {}
    for red_team, stats in red_teams.items():
        scores = stats["danger_scores"]
        target_vs_redteam_final[target][red_team] = {
            "total_tests": stats["total_tests"],
            "labels": dict(stats["labels"]),
            "avg_danger_score": sum(scores) / len(scores) if scores else 0
        }

# Aggregate by target model vs judge
target_vs_judge = defaultdict(lambda: defaultdict(lambda: {
    "total_tests": 0,
    "labels": Counter(),
    "danger_scores": []
}))

for result in all_results:
    target = result.get("target_model")
    judge = result.get("judge_model")
    label = result.get("label", "safe_refusal")
    danger_score = result.get("danger_score", 0)
    
    target_vs_judge[target][judge]["total_tests"] += 1
    target_vs_judge[target][judge]["labels"][label] += 1
    target_vs_judge[target][judge]["danger_scores"].append(danger_score)

# Convert to final format
target_vs_judge_final = {}
for target, judges in target_vs_judge.items():
    target_vs_judge_final[target] = {}
    for judge, stats in judges.items():
        scores = stats["danger_scores"]
        target_vs_judge_final[target][judge] = {
            "total_tests": stats["total_tests"],
            "labels": dict(stats["labels"]),
            "avg_danger_score": sum(scores) / len(scores) if scores else 0
        }

# Aggregate by (red_team, judge, target) triples for 3D heatmap
triple_agg = defaultdict(lambda: {
    "total_tests": 0,
    "labels": Counter(),
    "danger_scores": []
})

for result in all_results:
    target = result.get("target_model")
    red_team = result.get("red_team_model")
    judge = result.get("judge_model")
    label = result.get("label", "safe_refusal")
    danger_score = result.get("danger_score", 0)
    
    key = (red_team, judge, target)
    triple_agg[key]["total_tests"] += 1
    triple_agg[key]["labels"][label] += 1
    triple_agg[key]["danger_scores"].append(danger_score)

# Convert to flat array
triple_data = []
for (red_team, judge, target), stats in triple_agg.items():
    scores = stats["danger_scores"]
    triple_data.append({
        "red_team": red_team,
        "judge": judge,
        "target": target,
        "total_tests": stats["total_tests"],
        "labels": dict(stats["labels"]),
        "avg_danger_score": sum(scores) / len(scores) if scores else 0
    })

print(f"Generated {len(triple_data)} (red_team, judge, target) triples for 3D heatmap")

# Create enhanced master summary
master_summary = {
    "total_pairings": len(all_pairings),
    "models_tested": ALL_MODELS,
    "attack_rounds_per_pairing": 30,
    "max_prompt_retries": 3,
    "max_test_retries": 3,
    "json_translator_model": "llama3.1:8b",
    "pairings": pairing_summaries,
    "target_model_vs_red_team": target_vs_redteam_final,
    "target_model_vs_judge": target_vs_judge_final,
    "triple_data": triple_data,
    "generated_at": datetime.now(UTC).isoformat(),
}

# Save enhanced master summary
master_summary_file = results_dir / "master_summary.json"
with open(master_summary_file, "w", encoding="utf-8") as f:
    json.dump(master_summary, f, indent=2)

print(f"✓ Enhanced master summary saved to {master_summary_file}")
print(f"✓ Added target_model_vs_red_team groupings for {len(target_vs_redteam_final)} target models")
print(f"✓ Added target_model_vs_judge groupings for {len(target_vs_judge_final)} target models")
print("=" * 80)
