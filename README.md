# EECE-490-Hackathon

By; Karim Zarzour, Karim Hajj Ali, Tarek El Mourad

Bankruptcy Risk (Next-Year) — ML Predictor

Predict whether a public company will be failed next year (t+1) from its year-t financial features (X1..X18).
The project includes training, a Gradio GUI for manual inputs, and batch evaluation for CSV/Excel files.

Overview

Target (next-year event)

Preferred: Y_from_alive = 1 if status_label(t)=alive and status_label(t+1)=failed (onset).

Fallback: Y_any_next_fail = 1 if status_label(t+1)=failed (from any state).
The trainer picks Y_from_alive if positives exist; otherwise uses Y_any_next_fail.

Time split: Train 1999–2011, Val 2012–2014, Test 2015–2018 (prevents leakage).

Model: CatBoost (class-imbalance aware).
Threshold chosen on the validation set (F1-optimal); you can adjust it in the GUI.

Dataset schema

Required columns in american_bankruptcy.csv:

IDs: company_name, year, status_label in {alive, failed}

Features: X1..X18
X1 Current assets, X2 COGS, X3 Depreciation & amortization, X4 EBITDA, X5 Inventory, X6 Net Income,
X7 Total Receivables, X8 Market value, X9 Net sales, X10 Total assets, X11 Total Long-term debt,
X12 EBIT, X13 Gross Profit, X14 Total Current Liabilities, X15 Retained Earnings,
X16 Total Revenue, X17 Total Liabilities, X18 Total Operating Expenses.

Rows without a known t+1 record are excluded from training/evaluation.

Project structure
.
├── src/bankruptcy/
│   ├── data.py           # load CSV, build next-year labels, time split, arrays
│   ├── model.py          # CatBoost training, metrics, artifact saving
│   ├── thresholds.py     # F1-opt + cost-sensitive threshold helpers
│   └── app.py            # Gradio UI builder (manual X1..X18 + threshold)
├── scripts/
│   ├── train.py          # CLI: train + save model + metadata
│   ├── serve.py          # CLI: serve GUI (manual inputs)
│   └── evaluate.py       # CLI: batch evaluation + scored output
├── models/               # (gitignored) trained artifacts (.cbm, metadata.json)
├── data/                 # (you place input files here)
├── Dockerfile
├── docker/entrypoint.sh  # container entrypoint (train/serve/evaluate)
├── docker-compose.yml
├── requirements.txt
├── README.md
└── scored.xlsx           # contains predictions from uploaded dataset


Quickstart (no Docker)
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train (expects .\data\american_bankruptcy.csv)
python -m scripts.train --csv .\data\american_bankruptcy.csv --outdir models

# Serve GUI at http://localhost:7860
python -m scripts.serve --model models\catboost_bankruptcy.cbm --meta models\metadata.json

# Batch evaluate (scores CSV/XLSX and writes output)
python -m scripts.evaluate --input .\data\american_bankruptcy.csv --model models\catboost_bankruptcy.cbm --meta models\metadata.json --out .\scored.xlsx

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m scripts.train --csv ./data/american_bankruptcy.csv --outdir models
python -m scripts.serve --model models/catboost_bankruptcy.cbm --meta models/metadata.json
python -m scripts.evaluate --input ./data/american_bankruptcy.csv --model models/catboost_bankruptcy.cbm --meta models/metadata.json --out ./scored.xlsx

Docker usage

If Docker Desktop won’t start (virtualization off), run the no-Docker path above.
Otherwise:

Build the image
docker build -t bankruptcy-predictor:latest .

Train
$PWDPath = (Get-Location).Path
docker run --rm `
  -v "$PWDPath\data:/data:ro" `
  -v "$PWDPath\models:/models" `
  bankruptcy-predictor:latest `
  train --csv /data/american_bankruptcy.csv --outdir /models

Serve GUI (http://localhost:7860
)
$PWDPath = (Get-Location).Path
docker run --rm -p 7860:7860 `
  -v "$PWDPath\models:/models" `
  bankruptcy-predictor:latest `
  serve --model /models/catboost_bankruptcy.cbm --meta /models/metadata.json

Batch evaluate
$PWDPath = (Get-Location).Path
docker run --rm `
  -v "$PWDPath\data:/data" `
  -v "$PWDPath:/out" `
  -v "$PWDPath\models:/models" `
  bankruptcy-predictor:latest `
  evaluate --input /data/american_bankruptcy.csv --model /models/catboost_bankruptcy.cbm --meta /models/metadata.json --out /out/scored.xlsx

If you hit “exec format error”

Your entrypoint may have Windows line endings. Either:

Use --entrypoint python and run -m scripts.train / serve / evaluate, or

Normalize in the Dockerfile:

COPY docker/entrypoint.sh /usr/local/bin/app
RUN sed -i 's/\r$//' /usr/local/bin/app && chmod +x /usr/local/bin/app
ENTRYPOINT ["app"]

docker-compose (optional)
# Train (reads ./data, writes ./models)
docker compose run --rm train
# Serve
docker compose up app

GUI (manual feature input)

The app shows:

Probability of “failed next year (t+1)”

Predicted label based on a threshold slider
Default threshold is the validation F1-optimal value stored in metadata.json.
You can change it in the UI or retrain to re-compute it.

Reproducible training artifacts

After training, you’ll have:

models/catboost_bankruptcy.cbm — the trained model

models/metadata.json — feature list, target used, default threshold(s), medians

The GUI pulls feature medians from metadata to prefill the number inputs.
