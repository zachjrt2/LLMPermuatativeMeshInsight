# Permutative Mesh Dashboard

This dashboard evaluates 90+ red-team and judge model pairings to analyze adversarial phrase patterns and model safety trends.

## 🚀 How to Run (Critical)

Because the dashboard dynamically loads pre-computed aggregate data (`agg_manifest.json`, `phrase_intelligence.json`), modern browsers will block these requests if you open `index.html` directly from your file system (`file:///` URL).

**To view the dashboard correctly:**

1.  Open your terminal in this directory.
2.  Start a local Python server:
    ```bash
    python -m http.server
    ```
3.  Open your browser and navigate to:
    [http://localhost:8000/pairing_results/index.html](http://localhost:8000/pairing_results/index.html)

## 📊 Navigation Features
- **Dashboard**: Overview of pass rates, leaderboards, and per-model danger scores.
- **Phrase Intel**: Detailed analysis of phrase volatility. Helps identify "controversial" prompts that confuse judge models.
- **Flow Graph**: Visualization of the eval pipeline.
- **Heatmaps**: Categorical heatmaps for fine-grained vulnerability analysis.
