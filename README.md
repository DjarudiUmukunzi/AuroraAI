# AuroraAI — MVP (14-day build)

A scoped-down version of the full AuroraAI plan, sized for ~1-2 hrs/day
over 14 days. Goal: one working end-to-end pipeline — live NOAA data ->
Kp forecast model -> LLM explanation -> dashboard — by August 1.

This is deliberately **not** the full plan (no Data Factory, no
Databricks/Delta Lake, no multi-agent orchestration, no Azure ML
registry, no computer vision). Those come later when you circle back
for the actual AI-103/AI-300 certifications.

## Setup (do this first, ~15 min)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then fill in an API key (Azure OpenAI or plain OpenAI)
```

## Pipeline

```bash
python data/fetch_noaa_data.py   # pulls live NOAA Kp/solar wind/alert data -> data/raw/*.csv
python models/train_model.py     # trains XGBoost Kp forecaster -> models/kp_forecast_model.pkl
python agent/explain_agent.py    # quick test of the explanation agent
streamlit run dashboard/app.py   # full dashboard: chart + forecast + explanation
```

## 14-Day Checklist

- [ ] **Days 1-2** — Azure OpenAI resource (or plain OpenAI key) created; `.env` filled in; `fetch_noaa_data.py` runs and produces CSVs in `data/raw/`
- [ ] **Days 3-5** — Confirm the Kp history CSV looks sane (plot it, sanity-check date range); handle any NOAA schema quirks
- [ ] **Days 6-7** — `train_model.py` runs; test MAE beats naive persistence baseline (~0.5-0.8 Kp units)
- [ ] **Days 8-10** — `explain_agent.py` produces a real (non-template) explanation; tune the prompt
- [ ] **Days 11-12** — `streamlit run dashboard/app.py` works end-to-end locally
- [ ] **Days 13-14** — README polish, short screen recording, buffer for whatever broke

## Known simplifications (revisit for the real Phase 0-5 plan)

- Data lands as local CSV, not ADLS Gen2 / medallion Lakehouse
- One XGBoost model, no Azure ML model registry or experiment tracking
- One agent call, not a Supervisor/Research/Forecasting/Reasoning multi-agent graph
- No Azure AI Search / vector index — bulletin text goes straight into the prompt
- No CI/CD, monitoring, or drift detection
- No computer vision / aurora imagery module

## Next steps after Aug 1

Once this MVP is live, work back through the original plan's Phase 0-5
at the suggested ~20-week pace — this MVP already proves the concept
end-to-end, so each phase becomes "harden and formalize this piece"
rather than starting from scratch.
