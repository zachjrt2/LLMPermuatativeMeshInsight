import glob
import json
import os
import math
from collections import defaultdict

def run_phrase_analysis():
    base_dir = r"c:\Users\tho01699\Desktop\qwen\DataSite\pairing_results"
    pattern = os.path.join(base_dir, "*_results.jsonl")
    files = glob.glob(pattern)
    
    # phrase -> list of danger scores
    phrase_data = defaultdict(list)
    
    print(f"Analyzing {len(files)} files...")
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        prompt = data.get("attack_prompt")
                        score = data.get("danger_score")
                        if prompt is not None and score is not None:
                            phrase_data[prompt].append(float(score))
                    except:
                        continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Calculate stats
    stats = []
    for prompt, scores in phrase_data.items():
        if not scores: continue
        n = len(scores)
        avg = sum(scores) / n
        variance = sum((x - avg) ** 2 for x in scores) / n
        std_dev = math.sqrt(variance)
        stats.append({
            "prompt": prompt,
            "avg": avg,
            "std_dev": std_dev,
            "max": max(scores),
            "min": min(scores),
            "count": n
        })

    # Sort by avg danger (Dangerous)
    stats.sort(key=lambda x: x["avg"], reverse=True)
    
    # Keyword analysis
    def get_keywords(prompts):
        words = []
        stop_words = {'a', 'an', 'the', 'is', 'are', 'to', 'for', 'in', 'of', 'and', 'or', 'with', 'as', 'your', 'my', 'i', 'you', 'it', 'this', 'that', 'can', 'imag', 'writ', 'be', 'how', 'what'}
        for p in prompts:
            # simple tokenize
            for w in p.lower().split():
                clean = "".join(filter(str.isalnum, w))
                if clean and len(clean) > 3 and clean not in stop_words:
                    words.append(clean)
        return Counter(words).most_common(15)

    from collections import Counter
    dangerous_prompts = [s["prompt"] for s in stats[:50]]
    safe_prompts = [s["prompt"] for s in sorted(stats, key=lambda x: x["avg"])[:50]]
    volatile_prompts = [s["prompt"] for s in sorted(stats, key=lambda x: x["std_dev"], reverse=True)[:50]]

    output = {
        "most_dangerous": stats[:30],
        "safest": sorted(stats, key=lambda x: x["avg"])[:30],
        "most_volatile": sorted(stats, key=lambda x: x["std_dev"], reverse=True)[:50],
        "most_stable_dangerous": [s for s in stats if s["avg"] > 7 and s["std_dev"] < 1][:30],
        "keywords": {
            "dangerous": get_keywords(dangerous_prompts),
            "safe": get_keywords(safe_prompts),
            "volatile": get_keywords(volatile_prompts)
        },
        "all_stats": stats # limited version for scatter plot
    }

    output_path = r"c:\Users\tho01699\Desktop\qwen\DataSite\phrase_intelligence.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        # Save a subset for the scatter plot to keep file size reasonable
        # Only prompts with at least some activity or interesting stats
        json.dump(output, f, indent=4)
    
    print(f"Analysis complete. Results written to {output_path}")

if __name__ == "__main__":
    run_phrase_analysis()
