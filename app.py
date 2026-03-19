import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from data_generator import (
    generate_wafer_defects, generate_lot_history,
    generate_classification_results, DEFECT_TYPES, DEFECT_COLORS,
    PROCESS_STEPS, WAFER_RADIUS, LOTS
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ADC Yield Intelligence Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp { background-color: #0a0e1a; color: #e2e8f0; }

.metric-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #38bdf8;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1;
}
.metric-delta {
    font-size: 12px;
    color: #22c55e;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.section-header {
    font-size: 13px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #38bdf8;
    font-family: 'IBM Plex Mono', monospace;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
}
.badge-high { background: #1a2e1a; color: #22c55e; border: 1px solid #22c55e55; }
.badge-med  { background: #2a1f0a; color: #f59e0b; border: 1px solid #f59e0b55; }
.badge-low  { background: #2a0f0f; color: #ef4444; border: 1px solid #ef444455; }

div[data-testid="stSidebar"] {
    background: #0d1424;
    border-right: 1px solid #1e293b;
}
div[data-testid="stSidebar"] .stSelectbox label,
div[data-testid="stSidebar"] .stSlider label,
div[data-testid="stSidebar"] .stMultiSelect label {
    color: #94a3b8 !important;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}
h1 { color: #f8fafc !important; font-weight: 700 !important; }
h2, h3 { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 ADC Platform")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["📊 Overview Dashboard", "🗺️ Wafer Map", "🤖 ML Classifier", "📈 Trend Analysis"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<div class="metric-label">SIMULATION CONTROLS</div>', unsafe_allow_html=True)
    seed = st.slider("Random Seed", 0, 100, 42)
    n_defects = st.slider("Defect Count (Wafer Map)", 10, 200, 80)
    selected_lot = st.selectbox("Lot Filter", ["All"] + LOTS)
    selected_step = st.selectbox("Process Step", ["All"] + PROCESS_STEPS)
    st.markdown("---")
    st.markdown('<div style="color:#475569;font-size:11px;font-family:IBM Plex Mono,monospace">PTM Yield Systems<br>ADC Intelligence v2.1<br>© 2025</div>', unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_lot_history():
    return generate_lot_history()

@st.cache_data
def load_clf_results():
    return generate_classification_results(400)

lot_df = load_lot_history()
clf_df = load_clf_results()

# Apply filters
filtered = lot_df.copy()
if selected_lot != "All":
    filtered = filtered[filtered["lot_id"] == selected_lot]
if selected_step != "All":
    filtered = filtered[filtered["process_step"] == selected_step]


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.markdown("# ADC Yield Intelligence Platform")
    st.markdown('<div style="color:#64748b;font-family:IBM Plex Mono,monospace;font-size:13px;margin-bottom:24px">Automated Defect Classification · Inline Analysis · Yield Learning</div>', unsafe_allow_html=True)

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    avg_yield = filtered["yield_pct"].mean()
    avg_density = filtered["defect_density"].mean()
    total_wafers = len(filtered)
    clf_acc = clf_df["correct"].mean() * 100

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Wafer Yield</div>
            <div class="metric-value">{avg_yield:.1f}%</div>
            <div class="metric-delta">▲ Target: 85.0%</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Defect Density</div>
            <div class="metric-value">{avg_density:.4f}</div>
            <div class="metric-delta">defects/cm²</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Wafers Analyzed</div>
            <div class="metric-value">{total_wafers}</div>
            <div class="metric-delta">▲ {len(filtered['lot_id'].unique())} Lots</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">ADC Accuracy</div>
            <div class="metric-value">{clf_acc:.1f}%</div>
            <div class="metric-delta">RF Classifier · 6 classes</div>
        </div>""", unsafe_allow_html=True)

    # Charts row
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown('<div class="section-header">Yield Trend by Lot</div>', unsafe_allow_html=True)
        lot_agg = filtered.groupby("lot_id")["yield_pct"].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lot_agg["lot_id"], y=lot_agg["yield_pct"],
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2),
            marker=dict(size=8, color="#38bdf8", line=dict(color="#0a0e1a", width=2)),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.07)",
            name="Yield %"
        ))
        fig.add_hline(y=85, line_dash="dash", line_color="#f59e0b",
                      annotation_text="Target 85%", annotation_font_color="#f59e0b")
        fig.update_layout(
            plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(color="#94a3b8", family="IBM Plex Mono"),
            xaxis=dict(showgrid=False, tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(gridcolor="#1e293b", range=[50, 100]),
            margin=dict(l=0, r=0, t=10, b=0), height=280,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Defect Type Distribution</div>', unsafe_allow_html=True)
        defect_totals = {dt: filtered[f"count_{dt.lower()}"].sum() for dt in DEFECT_TYPES}
        fig2 = go.Figure(go.Pie(
            labels=list(defect_totals.keys()),
            values=list(defect_totals.values()),
            hole=0.55,
            marker=dict(colors=list(DEFECT_COLORS.values()),
                        line=dict(color="#0a0e1a", width=2)),
            textfont=dict(family="IBM Plex Mono", size=11),
        ))
        fig2.update_layout(
            plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(color="#94a3b8", family="IBM Plex Mono"),
            margin=dict(l=0, r=0, t=10, b=0), height=280,
            legend=dict(font=dict(size=10)),
            annotations=[dict(text="Defects", x=0.5, y=0.5,
                              font=dict(size=13, color="#e2e8f0",
                                        family="IBM Plex Mono"), showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Process step heatmap
    st.markdown('<div class="section-header">Defect Density by Process Step × Lot</div>', unsafe_allow_html=True)
    pivot = filtered.groupby(["process_step", "lot_id"])["defect_density"].mean().unstack(fill_value=0)
    fig3 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#0a0e1a"], [0.4, "#1e3a5f"], [0.7, "#1d4ed8"], [1, "#ef4444"]],
        showscale=True,
        colorbar=dict(tickfont=dict(family="IBM Plex Mono", size=10, color="#94a3b8")),
    ))
    fig3.update_layout(
        plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
        font=dict(color="#94a3b8", family="IBM Plex Mono"),
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=0, r=0, t=10, b=0), height=220,
    )
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — WAFER MAP
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Wafer Map":
    st.markdown("# Wafer Defect Map")
    st.markdown('<div style="color:#64748b;font-family:IBM Plex Mono,monospace;font-size:13px;margin-bottom:24px">Spatial defect visualization · ADC classification overlay</div>', unsafe_allow_html=True)

    col_ctrl, col_map = st.columns([1, 3])
    with col_ctrl:
        highlight = st.multiselect("Highlight Defect Types", DEFECT_TYPES, default=DEFECT_TYPES)
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)
        show_density = st.checkbox("Show Density Ring", value=True)

    wafer_df = generate_wafer_defects(n_defects=n_defects, seed=seed)
    wafer_filtered = wafer_df[
        (wafer_df["defect_type"].isin(highlight)) &
        (wafer_df["confidence"] >= min_conf)
    ]

    with col_map:
        fig = go.Figure()

        # Wafer circle
        theta = np.linspace(0, 2 * np.pi, 300)
        fig.add_trace(go.Scatter(
            x=WAFER_RADIUS * np.cos(theta),
            y=WAFER_RADIUS * np.sin(theta),
            mode="lines", line=dict(color="#334155", width=2),
            showlegend=False, hoverinfo="skip"
        ))
        if show_density:
            for r in [50, 100]:
                fig.add_trace(go.Scatter(
                    x=r * np.cos(theta), y=r * np.sin(theta),
                    mode="lines", line=dict(color="#1e293b", width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip"
                ))

        # Defect scatter per type
        for dt in highlight:
            sub = wafer_filtered[wafer_filtered["defect_type"] == dt]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"],
                mode="markers",
                name=dt,
                marker=dict(
                    size=sub["size_um"].clip(3, 16),
                    color=DEFECT_COLORS[dt],
                    opacity=sub["confidence"].clip(0.4, 1.0),
                    line=dict(color="#0a0e1a", width=1),
                ),
                hovertemplate=(
                    f"<b>{dt}</b><br>"
                    "X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>"
                    "Size: %{customdata[0]:.2f} µm<br>Conf: %{customdata[1]:.0%}"
                    "<extra></extra>"
                ),
                customdata=sub[["size_um", "confidence"]].values,
            ))

        fig.update_layout(
            plot_bgcolor="#090d1a", paper_bgcolor="#0a0e1a",
            xaxis=dict(showgrid=False, zeroline=False, range=[-175, 175],
                       scaleanchor="y", title="X (mm)",
                       titlefont=dict(family="IBM Plex Mono", color="#64748b")),
            yaxis=dict(showgrid=False, zeroline=False, range=[-175, 175],
                       title="Y (mm)",
                       titlefont=dict(family="IBM Plex Mono", color="#64748b")),
            font=dict(color="#94a3b8", family="IBM Plex Mono"),
            legend=dict(bgcolor="#0d1424", bordercolor="#1e293b",
                        borderwidth=1, font=dict(size=11)),
            margin=dict(l=0, r=0, t=10, b=0),
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Stats below map
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Defects Shown</div>
            <div class="metric-value">{len(wafer_filtered)}</div>
            <div class="metric-delta">of {len(wafer_df)} total</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        density = len(wafer_df) / (np.pi * (WAFER_RADIUS / 10) ** 2)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Defect Density</div>
            <div class="metric-value">{density:.4f}</div>
            <div class="metric-delta">defects/cm²</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        top_type = wafer_df["defect_type"].value_counts().idxmax()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Dominant Defect</div>
            <div class="metric-value" style="font-size:20px">{top_type}</div>
            <div class="metric-delta">{wafer_df["defect_type"].value_counts().max()} occurrences</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Defect Log Table</div>', unsafe_allow_html=True)
    display_df = wafer_filtered.copy()
    display_df.columns = ["X (mm)", "Y (mm)", "Defect Type", "Size (µm)", "Confidence"]
    display_df = display_df.round(3)
    st.dataframe(display_df, use_container_width=True, height=220,
                 hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Classifier":
    st.markdown("# ML Defect Classifier")
    st.markdown('<div style="color:#64748b;font-family:IBM Plex Mono,monospace;font-size:13px;margin-bottom:24px">Random Forest · 6-class ADC · Simulated Feature Extraction</div>', unsafe_allow_html=True)

    @st.cache_data
    def train_model():
        np.random.seed(42)
        n = 1000
        labels = np.random.choice(DEFECT_TYPES, n, p=[0.25, 0.30, 0.15, 0.10, 0.12, 0.08])
        label_idx = {d: i for i, d in enumerate(DEFECT_TYPES)}

        features = []
        for label in labels:
            li = label_idx[label]
            base = np.zeros(8)
            base[li] = np.random.uniform(0.6, 1.0)
            base += np.random.normal(0, 0.15, 8)
            features.append(base)

        X = np.array(features)
        y = labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return clf, X_test, y_test, y_pred

    clf_model, X_test, y_test, y_pred = train_model()

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred, labels=DEFECT_TYPES)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig_cm = go.Figure(go.Heatmap(
            z=cm_norm, x=DEFECT_TYPES, y=DEFECT_TYPES,
            colorscale=[[0, "#0a0e1a"], [0.5, "#1e3a8a"], [1, "#38bdf8"]],
            text=cm, texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono", size=11),
            showscale=True,
            colorbar=dict(tickfont=dict(family="IBM Plex Mono", size=9, color="#94a3b8")),
        ))
        fig_cm.update_layout(
            plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(color="#94a3b8", family="IBM Plex Mono"),
            xaxis=dict(title="Predicted", tickangle=30),
            yaxis=dict(title="Actual"),
            margin=dict(l=0, r=0, t=10, b=0), height=320,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Per-Class Precision / Recall</div>', unsafe_allow_html=True)
        report = classification_report(y_test, y_pred, labels=DEFECT_TYPES, output_dict=True)
        classes = DEFECT_TYPES
        precision = [report[c]["precision"] for c in classes]
        recall = [report[c]["recall"] for c in classes]

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Bar(name="Precision", x=classes, y=precision,
                                marker_color="#38bdf8", opacity=0.85))
        fig_pr.add_trace(go.Bar(name="Recall", x=classes, y=recall,
                                marker_color="#8b5cf6", opacity=0.85))
        fig_pr.update_layout(
            plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(color="#94a3b8", family="IBM Plex Mono"),
            barmode="group",
            yaxis=dict(gridcolor="#1e293b", range=[0, 1.1]),
            xaxis=dict(showgrid=False),
            legend=dict(bgcolor="#0d1424", bordercolor="#1e293b", borderwidth=1),
            margin=dict(l=0, r=0, t=10, b=0), height=320,
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # Feature importance
    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
    feat_names = [f"Feature_{i+1}" for i in range(8)]
    importances = clf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig_fi = go.Figure(go.Bar(
        x=[feat_names[i] for i in sorted_idx],
        y=[importances[i] for i in sorted_idx],
        marker=dict(
            color=importances[sorted_idx],
            colorscale=[[0, "#1e293b"], [1, "#38bdf8"]],
            showscale=False,
        ),
    ))
    fig_fi.update_layout(
        plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
        font=dict(color="#94a3b8", family="IBM Plex Mono"),
        yaxis=dict(gridcolor="#1e293b"),
        xaxis=dict(showgrid=False),
        margin=dict(l=0, r=0, t=10, b=0), height=220,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    overall_acc = (np.array(y_pred) == np.array(y_test)).mean() * 100
    st.markdown(f"""<div class="metric-card" style="text-align:center">
        <div class="metric-label">Overall Classifier Accuracy</div>
        <div class="metric-value" style="color:#22c55e">{overall_acc:.1f}%</div>
        <div class="metric-delta">RandomForest · 100 estimators · 750 train / 250 test</div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Trend Analysis":
    st.markdown("# Yield Trend Analysis")
    st.markdown('<div style="color:#64748b;font-family:IBM Plex Mono,monospace;font-size:13px;margin-bottom:24px">Longitudinal defect and yield analytics · Pareto analysis</div>', unsafe_allow_html=True)

    # Yield over time
    st.markdown('<div class="section-header">Yield % Over Time (by Process Step)</div>', unsafe_allow_html=True)
    time_agg = lot_df.groupby(["date", "process_step"])["yield_pct"].mean().reset_index()
    fig_t = px.line(
        time_agg, x="date", y="yield_pct", color="process_step",
        color_discrete_sequence=["#38bdf8", "#f97316", "#22c55e", "#8b5cf6", "#f59e0b", "#ef4444"],
    )
    fig_t.update_layout(
        plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
        font=dict(color="#94a3b8", family="IBM Plex Mono"),
        yaxis=dict(gridcolor="#1e293b", range=[40, 100]),
        xaxis=dict(showgrid=False),
        legend=dict(bgcolor="#0d1424", bordercolor="#1e293b", borderwidth=1, font=dict(size=10)),
        margin=dict(l=0, r=0, t=10, b=0), height=280,
    )
    st.plotly_chart(fig_t, use_container_width=True)

    col_p, col_b = st.columns(2)
    with col_p:
        st.markdown('<div class="section-header">Defect Pareto</div>', unsafe_allow_html=True)
        totals = {dt: lot_df[f"count_{dt.lower()}"].sum() for dt in DEFECT_TYPES}
        sorted_types = sorted(totals, key=totals.get, reverse=True)
        vals = [totals[dt] for dt in sorted_types]
        cumulative = np.cumsum(vals) / sum(vals) * 100

        fig_par = make_subplots(specs=[[{"secondary_y": True}]])
        fig_par.add_trace(go.Bar(
            x=sorted_types, y=vals,
            marker_color=[DEFECT_COLORS[dt] for dt in sorted_types],
            name="Count", opacity=0.85,
        ), secondary_y=False)
        fig_par.add_trace(go.Scatter(
            x=sorted_types, y=cumulative,
            mode="lines+markers", line=dict(color="#f59e0b", width=2),
            marker=dict(size=7, color="#f59e0b"),
            name="Cumulative %",
        ), secondary_y=True)
        fig_par.update_layout(
            plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(color="#94a3b8", family="IBM Plex Mono"),
            yaxis=dict(gridcolor="#1e293b"),
            yaxis2=dict(range=[0, 110], tickformat=".0f", ticksuffix="%"),
            xaxis=dict(showgrid=False),
            legend=dict(bgcolor="#0d1424", bordercolor="#1e293b", borderwidth=1, font=dict(size=10)),
            margin=dict(l=0, r=0, t=10, b=0), height=300,
        )
        st.plotly_chart(fig_par, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Yield vs Defect Density</div>', unsafe_allow_html=True)
        fig_sc = go.Figure(go.Scatter(
            x=lot_df["defect_density"], y=lot_df["yield_pct"],
            mode="markers",
            marker=dict(
                size=6, color=lot_df["yield_pct"],
                colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
                showscale=True, opacity=0.6,
                colorbar=dict(title="Yield %", tickfont=dict(family="IBM Plex Mono", size=9, color="#94a3b8")),
            ),
        ))
        fig_sc.update_layout(
            plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(color="#94a3b8", family="IBM Plex Mono"),
            xaxis=dict(gridcolor="#1e293b", title="Defect Density (defects/cm²)"),
            yaxis=dict(gridcolor="#1e293b", title="Yield %"),
            margin=dict(l=0, r=0, t=10, b=0), height=300,
        )
        st.plotly_chart(fig_sc, use_container_width=True)
