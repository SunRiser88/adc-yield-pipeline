# 🔬 ADC Yield Intelligence Platform
### Automated Defect Classification · Inline Analysis · Yield Learning

A portfolio-grade Render application simulating an **Automated Defect Classification (ADC)** pipeline for semiconductor wafer manufacturing — modeled after real-world tools like Klarity and Exensio.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Features

| Page | Description |
|------|-------------|
| **Overview Dashboard** | KPIs, yield trends, defect distribution, process-step heatmap |
| **Wafer Map** | Interactive spatial defect visualization with confidence filtering |
| **ML Classifier** | Random Forest ADC model with confusion matrix & per-class metrics |
| **Trend Analysis** | Longitudinal yield trends, Pareto chart, yield vs density scatter |

---

## 🏗️ Architecture

```
project1_adc_pipeline/
├── app.py               # Streamlit UI — 4 pages
├── data_generator.py    # Synthetic wafer + lot + classifier data
└── requirements.txt
```

**Data pipeline flow:**
```
Synthetic Wafer Scan → Defect Detection → Feature Extraction → RF Classifier → ADC Labels → Dashboard
```

---

## 🛠️ Tech Stack
- **Python** · **Render** · **Plotly** · **scikit-learn** · **pandas** · **numpy**

---
