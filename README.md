# 🚀 GNN Trading System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://github.com/psf/black)

**Advanced Graph Neural Network Trading System for Korean Financial Markets**

*End-to-end ML pipeline for stock and bond futures prediction using temporal graph attention networks*

[🎯 Features](#-key-features) • [🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🔧 Configuration](#-configuration)

</div>

---

## 🎯 Key Features

### 🧠 **Intelligent Architecture**
- **Temporal Graph Attention Networks (TGAT)** for multi-asset return prediction
- **Dynamic graph construction** with correlation and Granger causality edges
- **Real-time inference API** with FastAPI backend
- **Comprehensive backtesting engine** with realistic trading costs

### 📊 **Korean Market Specialization**
- **KRX/KTB futures integration** with intraday data support
- **Bank of Korea (ECOS) macroeconomic indicators** 
- **MIDAS transformation** for mixed-frequency data modeling
- **Won-denominated P&L calculations** with local market conventions

### 🔄 **Production-Ready Pipeline**
- **Phase 0-1**: Data ingestion & feature engineering
- **Phase 2**: Dynamic graph snapshot generation  
- **Phase 3**: TGAT model training with PyTorch Lightning
- **Phase 4**: Backtesting & real-time prediction deployment

---

## 📦 Installation

Choose your preferred package manager:

<details>
<summary><b>🐍 pip (Recommended)</b></summary>

```bash
# Clone and navigate to project
git clone <repository-url>
cd gnn_trading

# Install core dependencies
pip install -r requirements.txt

# Optional: Install development tools
pip install -r requirements-dev.txt
```
</details>

<details>
<summary><b>🐍 conda</b></summary>

```bash
# Clone and navigate to project  
git clone <repository-url>
cd gnn_trading

# Create and activate conda environment
conda env create -f environment.yml
conda activate gnn-trading
```
</details>

<details>
<summary><b>📦 Poetry</b></summary>

```bash
# Clone and navigate to project
git clone <repository-url>
cd gnn_trading

# Install with Poetry
poetry install
poetry shell
```
</details>

> **Requirements**: Python 3.11+ | Optional: CUDA-enabled GPU for faster training

---

## 🚀 Quick Start

### ⚡ **One-Click Pipeline Execution**

<details>
<summary><b>Using Poetry</b></summary>

```bash
# 🔄 Phase 1: Data Collection
poetry run python -m gnn_trading.data_pipeline.ingest \
    --start 2025-06-01 --end 2025-06-30

# 🛠️ Phase 2: Feature Engineering & Graph Construction
poetry run python -m gnn_trading.data_pipeline.feature_builder \
    --mkt_table market_intraday_202506 \
    --macro_table macro_indicators

poetry run python -m gnn_trading.graphs.graph_builder \
    --feature_root feature_store/processed

# 🧠 Phase 3: Model Training
poetry run python -m gnn_trading.models.trainer \
    --snapshot_dir graph_snapshots --epochs 50

# 📈 Phase 4A: Backtesting
poetry run python -m gnn_trading.backtest.engine \
    --config configs/backtest_config.yaml

# 🌐 Phase 4B: Real-time API (Optional)
poetry run uvicorn gnn_trading.api.main:app --port 8000 --reload
```
</details>

<details>
<summary><b>Using pip/conda</b></summary>

```bash
# 🔄 Phase 1: Data Collection
python -m gnn_trading.data_pipeline.ingest \
    --start 2025-06-01 --end 2025-06-30

# 🛠️ Phase 2: Feature Engineering & Graph Construction
python -m gnn_trading.data_pipeline.feature_builder \
    --mkt_table market_intraday_202506 \
    --macro_table macro_indicators

python -m gnn_trading.graphs.graph_builder \
    --feature_root feature_store/processed

# 🧠 Phase 3: Model Training
python -m gnn_trading.models.trainer \
    --snapshot_dir graph_snapshots --epochs 50

# 📈 Phase 4A: Backtesting
python -m gnn_trading.backtest.engine \
    --config configs/backtest_config.yaml

# 🌐 Phase 4B: Real-time API (Optional)
uvicorn gnn_trading.api.main:app --port 8000 --reload
```
</details>

> 🔑 **Important**: Configure your API keys and asset lists in `configs/source_config.yaml` before running

---

## 🏗️ Architecture Overview

```
📁 gnn_trading/
├── 📊 src/gnn_trading/
│   ├── 🔄 data_pipeline/     # Phase 0-1: Ingestion & Feature Engineering
│   ├── 🕸️ graphs/            # Phase 2: Dynamic Graph Construction
│   ├── 🧠 models/            # Phase 3: TGAT Training & Inference
│   ├── 📈 backtest/          # Phase 4A: Performance Evaluation
│   ├── 🌐 api/               # Phase 4B: Real-time Prediction API
│   └── 🛠️ utils/             # Utilities & Validation
├── 💾 feature_store/         # Parquet Data Lake (Auto-generated)
├── 📸 graph_snapshots/       # Temporal Graph Snapshots (.pt files)
├── 🎯 checkpoints/           # Trained Model Artifacts
├── ⚙️ configs/               # YAML Configuration Files
└── 📖 docs/                  # Detailed Documentation
```

---

## 🔧 Configuration

### **Core Configuration Files**

| File | Purpose | Key Settings |
|------|---------|-------------|
| 🔌 `source_config.yaml` | **Data Sources** | API endpoints, asset lists, credentials |
| 🛠️ `feature_config.yaml` | **Feature Engineering** | Resampling frequency, MIDAS parameters |
| 🕸️ `graph_config.yaml` | **Graph Construction** | Edge methods, correlation thresholds |
| 🧠 `train_config.yaml` | **Model Training** | Epochs, batch size, learning rate |
| 📈 `backtest_config.yaml` | **Backtesting** | Trading costs, slippage, risk metrics |

### **Example Configuration**

<details>
<summary><b>📊 Graph Configuration</b></summary>

```yaml
# configs/graph_config.yaml
snapshot_freq: '1D'           # Daily graph snapshots
corr_window: 60               # 60-minute correlation window
corr_threshold: 0.3           # Edge inclusion threshold
edge_method: 'corr'           # 'corr' or 'granger'
symbols: ['KS200', 'KTB3F']   # Target assets
indicators: ['CPI', 'BaseRate'] # Macro indicators
```
</details>

---

## 📊 Results & Performance

### **Backtesting Output**
- 📈 **P&L Analysis**: `backtest_result.csv` with cumulative returns and positions
- 📊 **Risk Metrics**: Sharpe ratio, Maximum Drawdown, Sortino ratio, VaR
- 📋 **Trade Statistics**: Win rate, average holding period, transaction costs

### **API Endpoints**
- `POST /predict` - Single prediction from graph snapshot
- `POST /batch_predict` - Batch predictions for multiple snapshots  
- `GET /health` - System health and model status
- `GET /model_info` - Model architecture and parameters
- `GET /docs` - Interactive API documentation

---

## 🔬 Advanced Features

### **📈 Financial Engineering**
- **MIDAS Transformation**: Mixed-frequency data modeling for macro indicators
- **Dynamic Risk Management**: Position sizing based on signal confidence
- **Transaction Cost Modeling**: Realistic slippage and commission simulation

### **🧠 Machine Learning**
- **Temporal Attention**: GRU-based sequence modeling with graph attention
- **Multi-asset Learning**: Joint prediction across correlated instruments  
- **Regularization**: Dropout, weight decay, and learning rate scheduling

### **🔧 Production Features**
- **Configuration Validation**: Automated config file validation
- **Data Quality Checks**: Parquet file integrity verification
- **Comprehensive Logging**: Structured logging with configurable levels
- **Error Handling**: Graceful failure handling with detailed error messages

---

## 🚀 Customization & Extension

### **💡 Quick Customization Tips**

1. **🔄 MIDAS Implementation**: Extend `feature_builder.fill_macro_daily()` with custom Almon polynomial weights
2. **🕸️ Edge Definitions**: Modify `graph_builder._edges_from_corr()` for custom relationship modeling  
3. **🎯 Loss Functions**: Implement custom loss functions in `models/tgat.py` for specific trading objectives
4. **📊 Risk Metrics**: Add custom performance metrics in `backtest/risk.py`

### **🏗️ Architecture Extensions**
- **Multi-timeframe Analysis**: Extend graph construction for multiple time horizons
- **Alternative Data Integration**: Add sentiment, news, or satellite data sources
- **Ensemble Methods**: Combine multiple TGAT models for robust predictions
- **Reinforcement Learning**: Integrate RL agents for dynamic position sizing

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **🍴 Fork** the repository
2. **🌟 Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **💾 Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **📤 Push** to the branch (`git push origin feature/amazing-feature`)
5. **🔄 Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch Geometric Team** for the excellent graph neural network framework
- **Bank of Korea** for providing comprehensive macroeconomic data APIs
- **Korea Exchange (KRX)** for market data infrastructure
- **Open Source Community** for the foundational libraries that made this project possible

---

<div align="center">

**Made with ❤️ for the Korean Financial Markets**

[⭐ Star this repo](../../stargazers) • [🐛 Report Bug](../../issues) • [💡 Request Feature](../../issues)

</div>
