---
title: BTC Autonomous Trading Platform
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
suggested_hardware: cpu-basic
app_port: 7860
pinned: false
---

# Bitcoin AI Market Prediction & Analytics Platform


A complete, production-oriented MVP platform that ingests Bitcoin market data, cleans and validates it, calculates features, trains XGBoost models, generates probabilistic predictions, backtests strategies, and exposes predictions through an API and a Next.js dashboard.

> **WARNING**: This system is for RESEARCH and PAPER TRADING only. It does not send live orders to any real exchange.

## Features
- **Historical & Live Data**: Modular Binance REST and WebSocket integration.
- **Data Quality**: Automated validation checks and machine-readable reports.
- **Feature Engineering**: Comprehensive technical indicators, perfectly aligned without look-ahead bias.
- **Configurable Targets**: Predicts probabilities for DOWN, NEUTRAL, UP.
- **Execution-Aware Backtesting**: Accurate slippage, fees, and signal shifting.
- **Live Worker**: Real-time feature calculation and prediction streaming.
- **Dashboard**: Professional Next.js application with Tailwind CSS and Recharts.

## Quickstart

### 1. Environment
```bash
cp .env.example .env
make setup
```

### 2. Run the Pipeline
```bash
make download     # Download historical data from Binance
make validate     # Generate data quality report
make features     # Build features and targets
make train        # Train XGBoost model and evaluate
make backtest     # Run backtester on test predictions
```

### 3. Run the Services
**Terminal 1:**
```bash
make api          # Starts FastAPI at http://localhost:8000
```
**Terminal 2:**
```bash
make worker       # Starts WebSocket ingestion and live predictions
```
**Terminal 3:**
```bash
make dashboard    # Starts Next.js app at http://localhost:3000
```

### 4. Docker Deployment
```bash
make docker-up
```

## Structure
- `src/config/`: Centralized settings.
- `src/data/`: Collectors and Validators.
- `src/features/`: Feature and Target Engineering.
- `src/models/`: Training, evaluation, and registry.
- `src/backtesting/`: Execution-aware backtesting engine.
- `src/api/` & `src/prediction/`: Services.
- `dashboard/`: Next.js UI.


Terminal 1 (API):

powershell
uvicorn src.api.main:app --reload
Terminal 2 (Worker):

powershell
python -m src.prediction.worker
Terminal 3 (Dashboard):

powershell
cd dashboard
npm run dev