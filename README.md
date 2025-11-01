# Thmanyah ML Customer Churn Project

## مقدمة
هذا المشروع يطوّر حلاً كاملاً للتنبؤ بانسحاب المشتركين (Customer Churn) لخدمة بث موسيقي تعتمد على نموذج الاشتراكات. يتم استخدام سجلات نشاط المستخدمين (event logs) لبناء خط معالجة بيانات، تدريب نموذج تعلم آلي، نشر واجهة برمجية للتنبؤ، مع أدوات للمراقبة وإعادة التدريب الدوري.

## Project Overview
- **Data ingestion & cleaning**: ingest Sparkify-like event logs (`customer_churn*.json`) and clean anomalies.
- **Feature engineering**: aggregate سلوك المستخدم خلال نافذة زمنية قابلة للضبط مع مؤشرات مثل عدد الأغاني، مدة الجلسات، التفاعلات الإيجابية والسلبية، ومستوى الاشتراك.
- **Model training & evaluation**: Gradient Boosting classifier (scikit-learn) مع تتبع التجارب بواسطة MLflow.
- **Serving**: FastAPI endpoint لتحويل أحداث المستخدم إلى احتمال انسحاب.
- **Retraining & monitoring**: مهام نصية لإعادة التدريب، اكتشاف انحراف البيانات، ومراقبة الأداء.
- **Tooling**: إدارة الحزم عبر `uv`, تنسيق الكود بـ `black`، ضبط الجودة بـ `ruff`, أتمتة باستخدام `Makefile`, `pre-commit`, وتهيئة Docker.

## Repository Layout
```
.
├── api/                     # FastAPI service
├── configs/                 # YAML configs for training & monitoring
├── docs/                    # Technical documentation
├── scripts/                 # CLI utilities (train, retrain, monitor)
├── src/churn_pipeline/      # Core package (config, features, trainer, monitoring)
├── tests/                   # Pytest-based regression tests
├── Makefile                 # Developer automation
├── Dockerfile               # Production container
└── README.md
```

## Getting Started
1. **Install uv** (once):
   ```bash
   pip install uv
   ```
2. **Sync dependencies** (system-wide virtual environment-free install):
   ```bash
   make setup
   ```
3. **Run tests & linters**:
   ```bash
   make lint
   make test
   ```
4. **Train the baseline model** (artifact stored in `artifacts/`):
   ```bash
   make train
   ```
5. **Serve the predictor**:
   ```bash
   make api
   ```
   Then interact with `POST /predict` using event payloads (see below).

> **Note**: The training scripts require `scikit-learn`, `joblib`, and `mlflow`. Ensure `make setup` completes successfully before running them.

## API Usage
`POST /predict`
```json
{
  "events": [
    {
      "ts": 1538352117000,
      "userId": "30",
      "sessionId": 29,
      "page": "NextSong",
      "level": "paid",
      "itemInSession": 50,
      "registration": 1538173362000,
      "gender": "M",
      "artist": "Martha Tilston",
      "song": "Rockpools",
      "length": 277.89016
    }
  ]
}
```
Response:
```json
{
  "churn_probability": 0.37,
  "churn_label": 0,
  "user_id": "30"
}
```

## Monitoring & Retraining
- **Retraining** on fresh data:
  ```bash
  python scripts/retrain.py --data path/to/new_events.json --register
  ```
- **Data/Concept drift** (requires baseline + new feature tables):
  ```bash
  python scripts/monitor.py --config configs/monitoring.yaml
  ```
Output includes PSI, KS statistics, and relative performance deltas. Thresholds are configurable.

## Exploratory Analysis
Generate descriptive charts and summaries:
```bash
make setup  # if not already installed
python scripts/analysis.py --data customer_churn_mini.json --output docs/charts
```
The script saves PNG figures (e.g., churn by subscription level, top pages for churners) and a JSON summary to `docs/charts/`. These assets can be embedded in reports or presentations.

## Docker Image
Build and run:
```bash
docker build -t churn-api .
docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts churn-api
```
Ensure `artifacts/model.joblib` exists before launching the container (via `make train`).

## Development Workflow
- Format: `make format`
- Activate pre-commit hooks: `pre-commit install`
- Run notebooks/analysis inside `docs/` (optional, not provided).

## Limitations & Next Steps
- Current feature builder relies on JSON event logs; consider integrating a streaming ingestion layer.
- Gradient Boosting is a strong baseline; experimenting with XGBoost/LightGBM might improve recall on the minority class.
- Monitoring script expects precomputed feature snapshots; automating extraction from production logs is future work.
- MLflow is optional; configure `MLFLOW_TRACKING_URI` to point to remote tracking servers when available.

## License
MIT License — see individual source files for details.

## Baseline Results
After running `make train` on `customer_churn_mini.json`, the held-out evaluation produced:

| Metric      | Value  |
|-------------|--------|
| Accuracy    | 0.78   |
| Precision   | 0.50   |
| Recall      | 0.44   |
| F1          | 0.47   |
| ROC-AUC     | 0.69   |
| PR-AUC      | 0.54   |

These metrics are also saved to `artifacts/metrics.json` for downstream monitoring.
