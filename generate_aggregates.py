"""
generate_aggregates.py
======================
Pre-aggregates all red-team JSONL result files into compact JSON files that
index.html can fetch without parsing megabytes of raw rows.

Run from:  DataSite/  (i.e., the parent of pairing_results/)
Output:    pairing_results/agg_*.json  (111 files total)
           pairing_results/agg_manifest.json

Usage:
    python generate_aggregates.py
    python generate_aggregates.py --out-dir pairing_results   # explicit path
"""

import json
import re
import sys
import glob
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

FILE_RE  = re.compile(r"^redteam-(.+?)_judge-(.+?)_results\.jsonl$")
ROUNDS_PER_FILE = 30

DANGER_BUCKETS = [
    ("None (0-1.4)",       0,   1.5),
    ("Minimal (1.5-2.4)", 1.5,  2.5),
    ("Very Low (2.5-3.4)",2.5,  3.5),
    ("Low (3.5-4.4)",     3.5,  4.5),
    ("Med-Low (4.5-5.4)", 4.5,  5.5),
    ("Medium (5.5-6.4)",  5.5,  6.5),
    ("Med-High (6.5-7.4)",6.5,  7.5),
    ("High (7.5-8.4)",    7.5,  8.5),
    ("Very High (8.5-9.4)",8.5, 9.5),
    ("Critical (9.5-10)", 9.5, 11),
]

def _bucket_idx(score):
    for i, (_, lo, hi) in enumerate(DANGER_BUCKETS):
        if lo <= score < hi:
            return i
    return len(DANGER_BUCKETS) - 1


def _normalize_label(label: str) -> str:
    """Return canonical one of: safe_refusal | partial_compliance | policy_violation"""
    l = (label or "").lower().strip()
    if l in ("safe_refusal", "safe refusal", "refusal"):
        return "safe_refusal"
    if l in ("partial_compliance", "partial compliance"):
        return "partial_compliance"
    return "policy_violation"


def load_jsonl(path: Path, round_offset: int = 0) -> list:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if round_offset:
                    row["round"] = (row.get("round") or 0) + round_offset
                rows.append(row)
            except json.JSONDecodeError:
                pass
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Core aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate(rows: list, file_count: int) -> dict:
    """Compute the full aggregate payload from a list of raw rows."""

    if not rows:
        return {}

    total = len(rows)
    models    = sorted({r["target_model"] for r in rows})
    categories = sorted({r.get("attack_category", "Unknown") for r in rows})

    # ── Label counts ─────────────────────────────────────────────────────────
    label_counts = defaultdict(int)
    for r in rows:
        label_counts[_normalize_label(r.get("label", ""))] += 1

    # ── Summary cards ────────────────────────────────────────────────────────
    all_rounds  = [r.get("round", 0) for r in rows]
    max_round   = max(all_rounds) if all_rounds else 0
    total_danger = sum(r.get("danger_score", 0) for r in rows)
    safe_count   = label_counts["safe_refusal"]

    summary_cards = {
        "total_rounds"   : max_round,
        "models_count"   : len(models),
        "avg_danger"     : round(total_danger / total, 2) if total else 0,
        "safety_rate_pct": round(safe_count / total * 100, 1) if total else 0,
    }

    # ── Per-model stats ───────────────────────────────────────────────────────
    model_stats = {}
    for model in models:
        mrs = [r for r in rows if r["target_model"] == model]
        m_safe = sum(1 for r in mrs if _normalize_label(r.get("label","")) == "safe_refusal")
        m_partial = sum(1 for r in mrs if _normalize_label(r.get("label","")) == "partial_compliance")
        m_viol    = sum(1 for r in mrs if _normalize_label(r.get("label","")) == "policy_violation")
        m_danger  = sum(r.get("danger_score", 0) for r in mrs)
        m_total   = len(mrs)
        model_stats[model] = {
            "total"          : m_total,
            "safe"           : m_safe,
            "partial"        : m_partial,
            "violation"      : m_viol,
            "avg_danger"     : round(m_danger / m_total, 2) if m_total else 0,
            "pass_rate_pct"  : round(m_safe / m_total * 100, 1) if m_total else 0,
        }

    # ── Per-category stats ────────────────────────────────────────────────────
    category_stats = {}
    for cat in categories:
        crs = [r for r in rows if r.get("attack_category") == cat]
        c_safe = sum(1 for r in crs if _normalize_label(r.get("label","")) == "safe_refusal")
        c_viol  = sum(1 for r in crs if _normalize_label(r.get("label","")) == "policy_violation")
        c_total = len(crs)
        c_danger = sum(r.get("danger_score", 0) for r in crs)
        category_stats[cat] = {
            "total"     : c_total,
            "safe"      : c_safe,
            "violation" : c_viol,
            "avg_danger": round(c_danger / c_total, 2) if c_total else 0,
            "pass_rate_pct": round(c_safe / c_total * 100, 1) if c_total else 0,
        }

    # ── Heatmap  (model × category → pass_rate, counts) ──────────────────────
    heatmap = {}
    for model in models:
        heatmap[model] = {}
        for cat in categories:
            mrs = [r for r in rows if r["target_model"] == model and r.get("attack_category") == cat]
            t = len(mrs)
            p = sum(1 for r in mrs if _normalize_label(r.get("label","")) == "safe_refusal")
            heatmap[model][cat] = {
                "total"    : t,
                "passed"   : p,
                "pass_rate": round(p / t, 4) if t else 0,
            }

    # ── Per-round cumulative trend (model → round → cumulative pass rate %) ──
    rounds_sorted = sorted({r.get("round", 0) for r in rows})
    trend = {}
    cumulative_danger = {}
    for model in models:
        trend[model] = {}
        cumulative_danger[model] = {}
        model_rows = [r for r in rows if r["target_model"] == model]
        for rnd in rounds_sorted:
            cum = [r for r in model_rows if r.get("round", 0) <= rnd]
            p   = sum(1 for r in cum if _normalize_label(r.get("label","")) == "safe_refusal")
            t   = len(cum)
            d   = sum(r.get("danger_score", 0) for r in cum)
            trend[model][str(rnd)]            = round(p / t * 100, 2) if t else None
            cumulative_danger[model][str(rnd)] = round(d / t, 3)   if t else None

    # ── Danger distribution (10 buckets) ─────────────────────────────────────
    danger_dist = [0] * len(DANGER_BUCKETS)
    for r in rows:
        danger_dist[_bucket_idx(r.get("danger_score", 0))] += 1

    return {
        "meta": {
            "total"       : total,
            "file_count"  : file_count,
            "models"      : models,
            "categories"  : categories,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary_cards"    : summary_cards,
        "label_counts"     : dict(label_counts),
        "model_stats"      : model_stats,
        "category_stats"   : category_stats,
        "heatmap"          : heatmap,
        "trend"            : trend,
        "cumulative_danger": cumulative_danger,
        "danger_dist"      : danger_dist,
        "danger_bucket_labels": [b[0] for b in DANGER_BUCKETS],
    }


def write_agg(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    kb = path.stat().st_size // 1024
    print(f"  [OK] {path.name}  ({kb} KB)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate pre-aggregated JSON files for index.html")
    parser.add_argument("--out-dir", default="pairing_results",
                        help="Directory containing the result JSONL files (default: pairing_results)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        print(f"ERROR: {out_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Discover all result JSONL files
    all_files = sorted(out_dir.glob("redteam-*_results.jsonl"))
    if not all_files:
        print("ERROR: No result JSONL files found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_files)} JSONL files in {out_dir}/\n")

    # Parse filenames ─────────────────────────────────────────────────────────
    parsed = []
    for f in all_files:
        m = FILE_RE.match(f.name)
        if m:
            parsed.append({"path": f, "rt": m.group(1), "judge": m.group(2)})

    rt_models    = sorted({p["rt"]    for p in parsed})
    judge_models = sorted({p["judge"] for p in parsed})

    manifest = {
        "master"   : "agg_master.json",
        "groups_rt": {},
        "groups_judge": {},
        "files"    : {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── 1. MASTER  (all 90 files combined) ───────────────────────────────────
    print("Building MASTER aggregate …")
    master_rows = []
    for i, item in enumerate(parsed):
        offset = i * ROUNDS_PER_FILE
        rows   = load_jsonl(item["path"], round_offset=offset)
        master_rows.extend(rows)
    agg = aggregate(master_rows, len(parsed))
    write_agg(out_dir / "agg_master.json", agg)

    # ── 2. PER-RT-MODEL groups ────────────────────────────────────────────────
    print("\nBuilding per-Red-Team-model aggregates …")
    for rt in rt_models:
        items = [p for p in parsed if p["rt"] == rt]
        rows  = []
        for i, item in enumerate(items):
            rows.extend(load_jsonl(item["path"], round_offset=i * ROUNDS_PER_FILE))
        slug = rt.replace(":", "_").replace("/", "_")
        fname = f"agg_group_rt_{slug}.json"
        manifest["groups_rt"][rt] = fname
        write_agg(out_dir / fname, aggregate(rows, len(items)))

    # ── 3. PER-JUDGE-MODEL groups ─────────────────────────────────────────────
    print("\nBuilding per-Judge-model aggregates …")
    for judge in judge_models:
        items = [p for p in parsed if p["judge"] == judge]
        rows  = []
        for i, item in enumerate(items):
            rows.extend(load_jsonl(item["path"], round_offset=i * ROUNDS_PER_FILE))
        slug = judge.replace(":", "_").replace("/", "_")
        fname = f"agg_group_judge_{slug}.json"
        manifest["groups_judge"][judge] = fname
        write_agg(out_dir / fname, aggregate(rows, len(items)))

    # ── 4. PER-FILE aggregates ────────────────────────────────────────────────
    print("\nBuilding per-file aggregates …")
    for item in parsed:
        rows  = load_jsonl(item["path"])
        slug  = item["path"].stem.replace("_results", "")  # redteam-rt_judge-j
        fname = f"agg_{slug}.json"
        manifest["files"][item["path"].name] = fname
        write_agg(out_dir / fname, aggregate(rows, 1))

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest_path = out_dir / "agg_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[OK] Manifest written -> {manifest_path}")

    total_files = 1 + len(rt_models) + len(judge_models) + len(parsed) + 1
    print(f"[OK] Done -- {total_files} files generated in {out_dir}/")


if __name__ == "__main__":
    main()
