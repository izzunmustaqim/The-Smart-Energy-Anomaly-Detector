# ⚡ Smart Energy Anomaly Detector

An intelligent energy monitoring system that uses **machine learning** to detect and explain anomalous power consumption patterns in household energy data.

<p align="center">
  <a href="https://the-smart-energy-anomaly-detector-p7ncfpgdbtmmwjgslvmnao.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker" alt="Docker">
</p>

> 🌐 **[Live Demo](https://the-smart-energy-anomaly-detector-p7ncfpgdbtmmwjgslvmnao.streamlit.app/)** — Try it now, no installation needed!

---

## 🎯 What It Does

Instead of just displaying energy numbers, this system **actively thinks and alerts**:

1. **Ingests** the UCI Individual Household Electric Power Consumption dataset (~2M readings over 47 months)
2. **Learns** the "normal" energy rhythm using dual ML models:
   - **Isolation Forest** — detects point anomalies in multivariate feature space
   - **Prophet** — forecasts expected consumption and flags deviations from seasonality patterns
3. **Explains** each anomaly in plain language: *"Tuesday 2pm spike of 8.4 kW is unusual — typical range is 2.1–3.8 kW. Water heater consumption is +140% above normal."*
4. **Visualizes** everything in a polished Streamlit dashboard with interactive charts

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                Docker Compose                    │
│                                                  │
│  ┌──────────────────┐   ┌─────────────────────┐ │
│  │   Processor       │   │   Streamlit App    │ │
│  │                    │   │                     │ │
│  │  UCI Download     │   │  📊 Dashboard       │ │
│  │  → Clean & Resample│  │  🧠 Smart Alerts   │ │
│  │  → Feature Eng.   │   │  🔬 Exploration    │ │
│  │  → IF + Prophet   │   │                     │ │
│  │  → Explain        │   │  Reads from shared  │ │
│  │  → Save to SQLite │   │  volumes            │ │
│  └────────┬──────────┘   └──────────┬──────────┘ │
│           │                          │            │
│           └──────┐    ┌──────────────┘            │
│                  ▼    ▼                           │
│         ┌─────────────────┐                      │
│         │  Shared Volumes  │                     │
│         │  data/ + db/     │                     │
│         └──────────────────┘                     │
└──────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- ~4 GB free disk space (dataset + Docker images)

### Run with Docker Compose

```bash
# Clone the repository
git clone <repo-url>
cd "The Smart Energy Anomaly Detector"

# Build and run everything
docker compose up --build

# The processor will:
#   1. Download the UCI dataset (~20 MB)
#   2. Clean and resample the data
#   3. Train Isolation Forest + Prophet models
#   4. Detect anomalies and generate explanations
#   5. Save results to SQLite

# Once the processor finishes, the Streamlit app launches at:
#   http://localhost:8501
```

### Run Locally (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements-processor.txt
pip install -r requirements-app.txt

# Run the data processor
python -m processor.main

# Start the Streamlit app
streamlit run app/main.py
```

---

## 📁 Project Structure

```
├── docker-compose.yml          # 2 services: processor + streamlit
├── Dockerfile.processor        # Batch data processing container
├── Dockerfile.app              # Streamlit frontend container
│
├── processor/                  # DATA PROCESSOR SERVICE
│   ├── main.py                 # Pipeline orchestrator
│   ├── config.py               # Pydantic settings (env vars)
│   ├── ingestion/              # Pluggable data source adapters
│   │   ├── base.py             # Abstract DataSource interface
│   │   ├── uci_adapter.py      # UCI dataset downloader/parser
│   │   └── csv_adapter.py      # Generic CSV adapter
│   ├── preprocessing/          # Data cleaning & feature engineering
│   │   ├── cleaner.py          # Missing values, resampling
│   │   └── features.py         # Cyclical time features, rolling stats
│   ├── models/                 # ML model implementations
│   │   ├── base.py             # Abstract AnomalyModel interface
│   │   ├── isolation_forest.py # Scikit-learn Isolation Forest
│   │   └── prophet_model.py    # Facebook Prophet forecasting
│   ├── detection/              # Anomaly detection orchestration
│   │   └── detector.py         # Weighted ensemble + thresholding
│   ├── explainability/         # Alert explanation engine
│   │   └── explainer.py        # Contextual natural-language explanations
│   └── storage/                # Persistence layer
│       └── db_manager.py       # SQLite CRUD operations
│
├── app/                        # STREAMLIT APP SERVICE
│   ├── main.py                 # Entry point with dark theme
│   ├── config.py               # App settings
│   ├── views/                  # Multi-page navigation
│   │   ├── dashboard.py        # KPIs, trends, heatmap, donut chart
│   │   ├── smart_alerts.py     # AI-explained anomaly cards
│   │   └── exploration.py      # Interactive data drill-down
│   ├── components/             # Reusable UI components
│   │   ├── charts.py           # Plotly chart builders
│   │   ├── alert_card.py       # Smart alert card renderer
│   │   └── filters.py          # Date/severity/sub-meter filters
│   └── services/               # Data access layer
│       └── data_service.py     # Cached reads from Parquet + SQLite
│
├── tests/                      # Test suite
├── data/                       # Shared volume (auto-populated)
│   ├── raw/                    # Downloaded UCI dataset
│   ├── processed/              # Cleaned Parquet files
│   └── models/                 # Serialized model artifacts
└── db/                         # SQLite database
    └── anomalies.db
```

---

## ⚙️ Configuration

All settings are configurable via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `RESAMPLE_FREQ` | `1h` | Resampling frequency (pandas offset alias) |
| `CONTAMINATION` | `0.02` | Expected anomaly fraction (0.0–0.5) |
| `IF_WEIGHT` | `0.5` | Isolation Forest weight in ensemble |
| `PROPHET_WEIGHT` | `0.5` | Prophet weight in ensemble |
| `ANOMALY_THRESHOLD_PERCENTILE` | `95` | Percentile cutoff for anomaly classification |

---

## 🧠 How the AI Works

### Isolation Forest
Trained on 16 multivariate features (power, voltage, intensity, sub-metering, cyclical time encodings, rolling statistics). Identifies data points that are **structurally different** from the majority.

### Prophet Forecasting
Learns daily, weekly, and yearly seasonality from the Global Active Power time series. Flags points whose **actual values deviate significantly** from the forecast using MAD-scaled residual analysis.

### Ensemble
Both models' scores are normalized to [0, 1] and combined with configurable weights. A percentile-based threshold (default: 95th percentile) determines the final anomaly classification.

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **📊 Dashboard** | KPI cards, 30-day trend with anomaly markers, hourly heatmap, sub-meter donut chart |
| **🧠 Smart Alerts** | Filterable, paginated anomaly cards with severity badges, contextual explanations, and mini context charts |
| **🔬 Exploration** | Interactive Plotly charts with zoom, sub-meter comparison, voltage stability, raw data table with CSV export |

---

## 📜 License

Dataset: [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235) — CC BY 4.0

---

## 🙏 Acknowledgements

- **Dataset**: Georges Hébrail & Alice Bérard, EDF R&D
- **ML**: scikit-learn, Prophet (Meta)
- **Visualization**: Streamlit, Plotly
