
import json
import os
from pathlib import Path
import sys

# Hardcoded from main.py
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

RESULTS_DIR = Path("pairing_results")

def check_missing():
    missing = []
    present = []
    
    for red in ALL_MODELS:
        for judge in ALL_MODELS:
            if red == judge:
                continue
            
            pairing_id = f"redteam-{red.replace(':', '-')}_judge-{judge.replace(':', '-')}"
            summary_file = RESULTS_DIR / f"{pairing_id}_summary.json"
            
            if not summary_file.exists():
                missing.append(pairing_id)
            else:
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if not data:
                            missing.append(f"{pairing_id} (EMPTY)")
                        else:
                            present.append(pairing_id)
                except Exception:
                    missing.append(f"{pairing_id} (CORRUPT)")
    
    print(f"Total Expected: {len(ALL_MODELS) * (len(ALL_MODELS) - 1)}")
    print(f"Present: {len(present)}")
    print(f"Missing: {len(missing)}")
    
    with open("missing_pairings.txt", "w", encoding="utf-8") as f:
        for m in missing:
            f.write(m + "\n")
            print(f"MISSING: {m}")
        
    return missing

if __name__ == "__main__":
    missing = check_missing()
