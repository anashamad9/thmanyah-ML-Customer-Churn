# Technical Report — Customer Churn Prediction

## 1. Problem Statement
Predict which subscribers of a music-streaming platform are likely to churn. Use historical event logs to learn engagement patterns and output per-user churn probabilities for targeted retention actions.

- **Business output**: a daily table or API that provides `user_id`, `churn_probability`, `risk_bucket`, and recommended next-best-action. (Current API returns probability and label; bucket/action mapping is planned.)
- **Operational definition of churn**: the user has a `page == "Cancellation Confirmation"` event. Users without this event are considered retained unless alternative rules are configured.

## 2. Data Understanding
### 2.1 Datasets
- `customer_churn.json` (كامل)
- `customer_churn_mini.json` (عينة مصغرة)
- One row equals one event (page view, song play, thumb, ad, etc.).

### 2.2 Core Fields
- **Identity**: `userId`, `sessionId`, `itemInSession`
- **Time**: `ts` (ms epoch), `registration` (ms epoch)
- **Page/action**: `page`, `method`, `status`
- **Content**: `artist`, `song`, `length`
- **User state**: `level ∈ {free, paid}`, `gender ∈ {M, F}`, `location`
- **Churn label**: derived from the definition above

### 2.3 Class Balance (mini dataset)
- Approximately **23 % churn**.
- Event model: many-to-one (multiple events per `userId`), sessions identified by `sessionId`, sorted by `ts`.

### 2.4 Data Quality Checks
1. Drop rows with missing, blank, or placeholder `userId` (e.g., `" "`, `"None"`).
2. Convert `ts` and `registration` to UTC datetimes from ms epoch.
3. Coerce numerics: `sessionId`, `itemInSession`, `length` → numeric; invalid entries coerced to `NaN` then handled downstream.
4. De-duplicate identical events (same `userId`, `sessionId`, `itemInSession`, `ts`).
5. Enforce categorical vocabularies: `level ∈ {free, paid}`; unexpected values mapped to `"unknown"`.
6. Validate monotonicity where relevant: ensure `registration ≤ first event timestamp` per user.

## 3. Exploratory Analysis
Run:
```bash
python scripts/analysis.py \
  --data customer_churn_mini.json \
  --output docs/charts
```

Key findings (see figures in `docs/charts/`):
- **Churn by level (`churn_rate_by_level.png`)**: free ≈ 0.32 vs. paid ≈ 0.18.
- **Engagement trend (`daily_engagement.png`)**: churners show a sharp pre-churn drop.
- **Top pages (`top_pages.png`)**: churners over-index on `Submit Downgrade` and `Cancellation Confirmation`; retained users on `NextSong` and social actions.
- **Feature gaps (`feature_differences.png`, `summary.json`)**: churners have fewer listening minutes (~2.48 k vs. ~2.76 k), fewer songs, shorter sessions.

**Design implication**: include intensity, diversity, session dynamics, and subscription-level transitions in the feature set.

## 4. Feature Engineering
### 4.1 Windowing
- Default snapshot window: last **30 days** of activity prior to reference time.
- For churners: exclude events at or after first `Cancellation Confirmation` to avoid leakage.
- For non-churners: reference time = dataset end or user’s final event.

### 4.2 User-Level Features
- **Volume**: `num_events`, `num_songs`, `total_listening_minutes`, `event_rate_per_hour`.
- **Sessions**: `num_sessions`, `avg_session_minutes`, `median_session_minutes`, `std_session_minutes`.
- **Interactions**: counts of `Thumbs Up`, `Thumbs Down`, `Add to Playlist`, `Add Friend`, `Roll Advert`.
- **Diversity**: `distinct_songs`, `distinct_artists`.
- **Subscription**: `paid_event_ratio`, last level (one-hot), downgrade frequency (future enhancement), time gap from downgrade to cancel (future).
- **Temporal**: `active_days`, potential `days_since_last_event` and `inactivity_streak_max` (roadmap).
- **Demographics**: gender one-hot, `account_age_days = snapshot_time - registration`.

### 4.3 Filtering
- Drop users with `< min_events_per_user` or `< min_sessions_per_user` (defaults 15 and 2 in config) to reduce sparsity noise.

### 4.4 Preprocessing
- Numeric features: median imputation + standard scaling.
- Categorical leftovers: impute most frequent; OHE handled inside pipeline.
- Date math performed in UTC.

## 5. Modeling
### 5.1 Algorithm
- `GradientBoostingClassifier` (scikit-learn) serves as the baseline.

### 5.2 Split Strategy
- Prefer temporal hold-out: last 14 days as test. Fallback to stratified split with `random_state=42` if insufficient history.

### 5.3 Pipeline
- `ColumnTransformer` with numeric pipeline (imputer → scaler) and categorical pipeline (imputer).
- Gradient Boosting appended as final estimator.
- Pipeline serialized via `joblib` to `artifacts/model.joblib`.

### 5.4 Hyperparameters (baseline)
- Current implementation uses scikit-learn defaults; recommended tuning: `n_estimators=300`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.8`, `random_state=42`.
- Consider probability calibration (`CalibratedClassifierCV`) if PR curve unstable.

### 5.5 Thresholding
- Default decision threshold `t = 0.5`.
- Provide cost-aware threshold `t*` that minimizes `λ_FN * FN + λ_FP * FP` once campaign economics are defined.

## 6. Evaluation
### 6.1 Metrics on Hold-out
- Accuracy: **0.78**
- Precision: **0.50**
- Recall: **0.44**
- F1-score: **0.47**
- ROC-AUC: **0.69**
- PR-AUC: **0.54**

Artifacts:
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png` (future work: add visualization)
- `artifacts/pr_curve.png`, `artifacts/roc_curve.png` (future work)

### 6.2 Interpretation
- Moderate separability (ROC-AUC 0.69).
- Threshold tuning can trade precision vs. recall for campaign goals.
- Track PR-AUC because of class imbalance.

### 6.3 Note on Comparability
- Report metrics with the exact data slice, window size, and seed. Do not compare across different windows or seeds without recalculating baselines.

## 7. Error Analysis
- **False positives**: highly engaged free users with many `Roll Advert` events flagged as churn. Add monetization propensity features and cost-sensitive training.
- **False negatives**: abrupt inactivity without explicit cancel events. Add `inactivity_streak_max`, rolling engagement deltas.
- **Ablations**: remove feature families (volume, sessions, interactions) to quantify contribution; log ΔAUC and ΔPR-AUC per family in future experiments.

## 8. Retraining Strategy
### 8.1 Script
```bash
python scripts/retrain.py \
  --data customer_churn_YYYYMM.json \
  --register
```
- Extend script with `--window-days`, `--track-mlflow` options as roadmap.

### 8.2 Cadence
- Weekly on fresh partitions. Roll the window forward; keep last *N* models for rollback.

### 8.3 Registry
- Save model to `models/` with semantic versioning (`churn-gb-vMAJOR.MINOR.PATCH.joblib`).
- Record feature schema hash, training dates, git commit, and data snapshot IDs (planned enhancements).

## 9. Monitoring
### 9.1 Data Drift
- Population Stability Index (PSI) and Kolmogorov–Smirnov (KS) by feature.
- Thresholds in `configs/monitoring.yaml`. Flag PSI > 0.2 or KS > 0.1 (p-value checks to be added).

### 9.2 Concept Drift
- Compare live PR-AUC and recall@`t*` to baseline. Alert on ≥10 % relative drop.

### 9.3 Serving Checks
- Input schema validator, percent missing per feature, out-of-range detectors (roadmap).
- Daily dashboard: users scored, risk bucket distribution, top drifting features.

### 9.4 Command
```bash
python scripts/monitor.py --config configs/monitoring.yaml
```
- Extend to accept explicit baseline/current score files (e.g., `--scores data/scores_YYYYMMDD.parquet`) in future iterations.

## 10. Packaging & Tooling
### 10.1 Project Layout
```
.
├─ api/
│  └─ main.py
├─ configs/
│  ├─ training.yaml
│  └─ monitoring.yaml
├─ scripts/
│  ├─ analysis.py
│  ├─ train.py
│  ├─ retrain.py
│  └─ monitor.py
├─ src/churn_pipeline/
├─ docs/charts/
├─ artifacts/
├─ models/
├─ tests/
├─ Dockerfile
├─ pyproject.toml
└─ Makefile
```

### 10.2 Dependencies
- Managed with `uv` via `pyproject.toml`.

### 10.3 Quality
- `ruff`, `black`, `pytest`, `pre-commit`.

### 10.4 Container
- `Dockerfile` runs FastAPI with `uvicorn`. Add `/health` route to API for readiness (todo).

### 10.5 Automation
```
make setup        # install dependencies
python scripts/analysis.py ...  # EDA (add make target if desired)
make train        # train on configured dataset
make retrain      # retrain with optional data override
make api          # run API locally
make monitor      # run monitoring checks
```

## 11. Inference API (FastAPI)
- `POST /predict`
  - Input: JSON with `events` list of raw user events (single user per payload).
  - Output (current): `{"user_id": "...", "churn_probability": 0.xx, "churn_label": 0/1}`
  - Roadmap: include `risk_bucket`, `recommended_action`, and SHAP-style contributions.
- `GET /health`
  - TODO: returns `{"status": "ok", "model_version": "churn-gb-v1.2.0"}` once implemented.
- Batch scoring: roadmap to accept JSON/Parquet uploads and provide signed URLs.

## 12. Governance, Privacy, and Security
- **PII handling**: hash `userId` at ingestion; store salts in secure vault.
- **Data retention**: raw events 180 days; feature tables 90 days; logs 30 days.
- **Access control**: principle of least privilege; read-only production data for scoring jobs.
- **Reproducibility**: pin dependencies, fix seeds, log config and git commit SHA.
- **Compliance**: document DPIA if applicable; support opt-out from profiling where required.

## 13. Challenges & Recommendations
- **Class imbalance**: continue monitoring PR-AUC; experiment with class weights, focal loss, or SMOTE on training data only.
- **Label uncertainty**: evaluate inactivity-based pseudo-labels as weak supervision.
- **Feature freshness**: consider a feature store for low-latency aggregates.
- **Model upgrades**:
  - Evaluate LightGBM/XGBoost and calibrated logistic regression baselines.
  - Add SHAP for global/per-user explanations.
  - Implement incremental training on rolling windows.
  - Wire monitoring to Prometheus/Grafana with alert thresholds.

## 14. Reproducible Commands
- **Train**:
  ```bash
  make train PYTHON=python3.11
  ```
- **Retrain with new data**:
  ```bash
  python scripts/retrain.py --data customer_churn_YYYYMM.json --register
  ```
- **Serve API**:
  ```bash
  make api PYTHON=python3.11
  ```
- **Run monitoring**:
  ```bash
  make monitor PYTHON=python3.11
  ```
- **Docker**:
  ```bash
  docker build -t churn-api:1.0 .
  docker run -p 8000:8000 churn-api:1.0
  ```

## 15. Configuration Examples
### 15.1 `configs/training.yaml`
```yaml
data_path: customer_churn_mini.json
feature_config:
  lookback_days: 30
  min_events_per_user: 15
  min_sessions_per_user: 2
split_config:
  method: temporal
  temporal_holdout_days: 14
target_column: churned
```

### 15.2 `configs/monitoring.yaml`
```yaml
baseline_features_path: artifacts/features.parquet
current_features_path: artifacts/current_features.parquet
psi_threshold: 0.2
ks_threshold: 0.1
performance_drop_threshold: 0.1
```

## 16. Deliverables
- **Code**: `src/`, `scripts/`, `api/`, `configs/`
- **Trained model**: `artifacts/model.joblib` (latest), optional registry copy under `models/`
- **Metrics and plots**: `artifacts/metrics.json`, (future) ROC/PR curves
- **EDA figures**: `docs/charts/`
- **API contract**: section 11 of this document
- **Runbook**: section 14

## 17. Open Items
1. Validate decision threshold `t*` with marketing economics.
2. Add inactivity-based label variant and compare performance.
3. Implement SHAP explanations and include in API responses.
4. Finalize retention action mapping per risk bucket.
5. Generate and store ROC/PR curve visualizations alongside metrics.
