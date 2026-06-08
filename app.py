import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os, pickle, datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="t-SNE · Dimensionality Reduction", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family:'Syne',sans-serif; background:#080b12; color:#dde1f0; }
.main .block-container { background:#080b12; padding:2rem 3rem; max-width:1400px; }
[data-testid="stSidebar"] { background:#0c0f1a !important; border-right:1px solid #1a2040; }
[data-testid="stSidebar"] .block-container { padding:1.4rem 1rem; }

.hero {
    background:linear-gradient(135deg,#0a0d1e 0%,#0f1428 60%,#080b18 100%);
    border:1px solid #1a3040; border-radius:14px;
    padding:2rem 2.5rem; margin-bottom:1.8rem; position:relative; overflow:hidden;
}
.hero::after {
    content:''; position:absolute; bottom:-80px; right:-40px;
    width:300px; height:300px;
    background:radial-gradient(circle,rgba(129,140,248,.09) 0%,transparent 70%);
    border-radius:50%;
}
.hero h1 {
    font-size:2.3rem; font-weight:800;
    background:linear-gradient(90deg,#818cf8 0%,#c084fc 45%,#f472b6 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0 0 .3rem 0; font-family:'JetBrains Mono',monospace; letter-spacing:-1px;
}
.hero p { color:#4a5580; font-size:.9rem; margin:0; }

.ct {
    font-size:.68rem; font-weight:700; letter-spacing:.14em;
    text-transform:uppercase; color:#818cf8; margin-bottom:.5rem;
    font-family:'JetBrains Mono',monospace;
}

.metric-row { display:flex; gap:.9rem; flex-wrap:wrap; margin-bottom:1.4rem; }
.metric-box {
    flex:1; min-width:120px; background:#0c0f1a; border:1px solid #1a2040;
    border-radius:10px; padding:1rem 1.2rem; text-align:center;
}
.metric-box .val {
    font-size:1.8rem; font-weight:700; font-family:'JetBrains Mono',monospace;
    background:linear-gradient(90deg,#818cf8,#c084fc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.metric-box .lbl { font-size:.68rem; color:#3a4560; text-transform:uppercase;
                   letter-spacing:.09em; margin-top:.2rem; }

.train-wrap {
    background:#0c0f1a; border:1px solid #1a3040; border-radius:10px;
    padding:1.1rem 1.6rem; margin:1.1rem 0; display:flex;
    align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
}
.train-info { color:#4a5580; font-size:.84rem; }
.train-info strong { color:#8892b0; }

.info-card {
    background:#0c0f1a; border:1px solid #1a2040; border-radius:10px;
    padding:1.1rem 1.5rem; margin-bottom:1.2rem; font-size:.82rem; color:#4a5580;
    line-height:1.6;
}
.info-card strong { color:#818cf8; }

.stButton > button {
    background:linear-gradient(135deg,#4f46e5,#9333ea) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-family:'JetBrains Mono',monospace !important; font-size:.82rem !important;
    font-weight:600 !important; letter-spacing:.05em !important;
    padding:.55rem 1.5rem !important; transition:opacity .2s !important;
}
.stButton > button:hover { opacity:.82 !important; }

[data-testid="stSlider"] > div > div > div { background:#818cf8 !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background-color:#0c0f1a !important; border-color:#1a2040 !important; color:#dde1f0 !important;
}
[data-testid="stFileUploader"] section {
    background:#0c0f1a !important; border:1px dashed #1a3040 !important; border-radius:10px !important;
}
[data-testid="stDataFrame"] { border:1px solid #1a2040; border-radius:8px; }
hr { border-color:#1a2040 !important; }

.saved-banner {
    background:#071318; border:1px solid #1a1040; border-radius:10px;
    padding:1rem 1.5rem; margin-top:1.5rem;
    font-family:'JetBrains Mono',monospace; font-size:.82rem; color:#818cf8;
}
.saved-banner span { color:#3a4560; }

.warn-box {
    background:#100a1a; border:1px solid #2a1040; border-radius:10px;
    padding:.9rem 1.3rem; margin:.8rem 0;
    font-size:.8rem; color:#c084fc; font-family:'JetBrains Mono',monospace;
}

.js-plotly-plot .plotly .modebar { background:transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Dirs ──────────────────────────────────────────────────────────────────────
RAW_DIR, OUT_DIR, MODEL_DIR = "data/raw", "data/tsne", "models"
for d in [RAW_DIR, OUT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key in ["tsne_embedding","tsne_X","tsne_feature_cols",
            "tsne_df_result","tsne_n","tsne_params",
            "tsne_model_path","tsne_out_csv","tsne_scaler"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Helpers ───────────────────────────────────────────────────────────────────
PALETTE = ["#818cf8","#c084fc","#f472b6","#2dd4bf","#34d399",
           "#fb923c","#facc15","#a78bfa","#4ade80","#f87171"]

def mpl_dark(figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#080b12")
    ax.set_facecolor("#0c0f1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2040")
    ax.tick_params(colors="#4a5580", labelsize=8)
    ax.xaxis.label.set_color("#4a5580"); ax.yaxis.label.set_color("#4a5580")
    ax.title.set_color("#c0c8e0")
    return fig, ax

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧬 t-SNE</h1>
  <p>t-Distributed Stochastic Neighbour Embedding · Cluster Visualisation · 2-D & 3-D · Interactive</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()
    st.markdown("**Dataset**")
    uploaded = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"],
                                label_visibility="collapsed")
    st.divider()
    st.markdown("**t-SNE Parameters**")
    n_components  = st.selectbox("Output dimensions", [2, 3], index=0)
    perplexity    = st.slider("Perplexity", 5, 100, 30,
                              help="Related to number of nearest neighbours. Typical values: 5–50.")
    learning_rate = st.slider("Learning rate", 10, 1000, 200,
                              help="Controls step size during optimisation.")
    n_iter        = st.slider("Max iterations", 250, 2000, 1000, step=50)
    early_exag    = st.slider("Early exaggeration", 4, 20, 12)
    st.divider()
    st.markdown("**Preprocessing**")
    scale_data    = st.checkbox("Standardise features", value=True)
    pca_init      = st.checkbox("PCA initialisation", value=True,
                                help="Initialise embedding from PCA; faster & more stable.")
    st.divider()
    st.markdown("**Colour by**")
    color_col_choice = st.text_input("Column name to colour scatter (optional)",
                                     placeholder="e.g. species, label …")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df_raw = None
if uploaded:
    raw_path = os.path.join(RAW_DIR, uploaded.name)
    with open(raw_path, "wb") as f: f.write(uploaded.getbuffer())
    df_raw = pd.read_csv(raw_path) if uploaded.name.endswith(".csv") \
             else pd.read_excel(raw_path)
    st.success(f"✔ **{uploaded.name}** — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")

if df_raw is None:
    st.info("⬆ Upload a CSV or Excel file from the sidebar to get started.")
    st.stop()

num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
if len(num_cols) < 2:
    st.error("Need at least 2 numeric columns."); st.stop()

with st.expander("🔍 Preview data", expanded=False):
    st.dataframe(df_raw.head(200), use_container_width=True)

feature_cols = st.multiselect("", options=num_cols, default=num_cols,
                              placeholder="Select feature columns…",
                              label_visibility="collapsed")
if len(feature_cols) < 2:
    st.warning("Select at least 2 feature columns."); st.stop()

# ── t-SNE notes ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="info-card">
  ℹ️ <strong>t-SNE notes:</strong>
  &nbsp;Perplexity must be &lt; number of samples. 
  &nbsp;Distances between clusters are <em>not</em> meaningful — only local structure is preserved. 
  &nbsp;Re-running with the same parameters may produce different orientations (non-deterministic). 
  &nbsp;For large datasets (&gt;5 k rows), consider downsampling first.
</div>
""", unsafe_allow_html=True)

df_clean     = df_raw[feature_cols].dropna()
n_samples    = len(df_clean)
perp_capped  = min(perplexity, n_samples - 1)

if perplexity != perp_capped:
    st.markdown(f'<div class="warn-box">⚠ Perplexity capped to {perp_capped} (must be &lt; n_samples={n_samples})</div>',
                unsafe_allow_html=True)

if n_samples > 5000:
    st.markdown(f'<div class="warn-box">⚠ {n_samples:,} samples detected — t-SNE may be slow. Consider enabling PCA init and reducing iterations.</div>',
                unsafe_allow_html=True)

X_raw  = df_clean.values
sc_tmp = StandardScaler()
X      = sc_tmp.fit_transform(X_raw) if scale_data else X_raw.astype(float)

# ── Fit button ────────────────────────────────────────────────────────────────
params_summary = (f"n_components={n_components}  ·  perplexity={perp_capped}  ·  "
                  f"lr={learning_rate}  ·  iter={n_iter}  ·  "
                  f"init={'pca' if pca_init else 'random'}  ·  scaled={scale_data}")

st.markdown(f"""
<div class="train-wrap">
  <div class="train-info">
    <strong>{n_samples:,}</strong> samples &nbsp;·&nbsp;
    <strong>{len(feature_cols)}</strong> features &nbsp;·&nbsp;
    {params_summary}
  </div>
""", unsafe_allow_html=True)
train_btn = st.button("🧬 Run t-SNE", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

# ── Fit ───────────────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner(f"Running t-SNE ({n_iter} iterations) … this may take a moment ⏳"):
        tsne = TSNE(
            n_components=n_components,
            perplexity=perp_capped,
            learning_rate=learning_rate,
            n_iter=n_iter,
            early_exaggeration=early_exag,
            init="pca" if pca_init else "random",
            random_state=42,
            verbose=0,
        )
        embedding = tsne.fit_transform(X)

    model_path = os.path.join(MODEL_DIR,
                              f"tsne_n{n_components}_p{perp_capped}_i{n_iter}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "embedding": embedding,
            "scaler": sc_tmp,
            "features": feature_cols,
            "scaled": scale_data,
            "params": dict(n_components=n_components, perplexity=perp_capped,
                           learning_rate=learning_rate, n_iter=n_iter,
                           early_exaggeration=early_exag,
                           init="pca" if pca_init else "random"),
            "kl_divergence": getattr(tsne, "kl_divergence_", None),
        }, f)

    valid_idx = df_raw[feature_cols].dropna().index
    df_result = df_raw.loc[valid_idx].copy()
    for i in range(n_components):
        df_result[f"TSNE{i+1}"] = np.round(embedding[:, i], 5)
    out_csv = os.path.join(OUT_DIR,
                           f"tsne_n{n_components}_p{perp_capped}_i{n_iter}.csv")
    df_result.to_csv(out_csv, index=False)

    st.session_state.tsne_embedding   = embedding
    st.session_state.tsne_X           = X
    st.session_state.tsne_feature_cols= feature_cols
    st.session_state.tsne_df_result   = df_result
    st.session_state.tsne_n           = n_components
    st.session_state.tsne_params      = dict(
        perplexity=perp_capped, learning_rate=learning_rate,
        n_iter=n_iter, kl=getattr(tsne, "kl_divergence_", None))
    st.session_state.tsne_model_path  = model_path
    st.session_state.tsne_out_csv     = out_csv
    st.session_state.tsne_scaler      = sc_tmp

# ── Guard ─────────────────────────────────────────────────────────────────────
if st.session_state.tsne_embedding is None:
    st.stop()

embedding    = st.session_state.tsne_embedding
X_ss         = st.session_state.tsne_X
feature_cols = st.session_state.tsne_feature_cols
df_result    = st.session_state.tsne_df_result
n_ss         = st.session_state.tsne_n
p            = st.session_state.tsne_params
model_path   = st.session_state.tsne_model_path
out_csv      = st.session_state.tsne_out_csv

st.divider()
st.markdown("### 📊 Results")

# ── Metrics ───────────────────────────────────────────────────────────────────
kl_str = f"{p['kl']:.4f}" if p.get("kl") is not None else "—"
st.markdown(f"""
<div class="metric-row">
  <div class="metric-box"><div class="val">{n_ss}</div><div class="lbl">Dimensions</div></div>
  <div class="metric-box"><div class="val">{p['perplexity']}</div><div class="lbl">Perplexity</div></div>
  <div class="metric-box"><div class="val">{p['learning_rate']}</div><div class="lbl">Learning Rate</div></div>
  <div class="metric-box"><div class="val">{p['n_iter']}</div><div class="lbl">Iterations</div></div>
  <div class="metric-box"><div class="val">{kl_str}</div><div class="lbl">KL Divergence</div></div>
  <div class="metric-box"><div class="val">{len(embedding):,}</div><div class="lbl">Samples</div></div>
  <div class="metric-box"><div class="val">{len(feature_cols)}</div><div class="lbl">Features</div></div>
</div>
""", unsafe_allow_html=True)

# ── Resolve colour column ──────────────────────────────────────────────────────
color_vals  = None
color_label = None
if color_col_choice and color_col_choice in df_result.columns:
    raw_cv = df_result[color_col_choice].values
    if pd.api.types.is_numeric_dtype(raw_cv):
        color_vals  = raw_cv.astype(float)
        color_label = color_col_choice
    else:
        cats        = pd.Categorical(raw_cv)
        color_vals  = cats.codes.astype(float)
        color_label = color_col_choice

# ══════════════════════════════════════════════════════════════════════════════
# VIZ 1 — 2-D scatter (always)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="ct">t-SNE 2-D Embedding Scatter</div>', unsafe_allow_html=True)

fig_2d, ax_2d = mpl_dark((12, 6))
if color_vals is not None:
    sc_plot = ax_2d.scatter(embedding[:, 0], embedding[:, 1],
                            c=color_vals, cmap="plasma",
                            s=12, alpha=0.70, linewidths=0)
    plt.colorbar(sc_plot, ax=ax_2d, label=color_label,
                 fraction=0.02, pad=0.01)
else:
    ax_2d.scatter(embedding[:, 0], embedding[:, 1],
                  color=PALETTE[0], s=12, alpha=0.55, linewidths=0)

ax_2d.axhline(0, color="#1a2040", lw=0.8)
ax_2d.axvline(0, color="#1a2040", lw=0.8)
ax_2d.set_xlabel("t-SNE 1"); ax_2d.set_ylabel("t-SNE 2")
ax_2d.set_title(f"t-SNE Embedding  (perplexity={p['perplexity']}  ·  iter={p['n_iter']})", fontsize=10)
plt.tight_layout(); st.pyplot(fig_2d); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# VIZ 2 — Density / point-count heatmap overlay
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="ct">Embedding Density Heatmap</div>', unsafe_allow_html=True)

fig_dens, ax_dens = mpl_dark((12, 5))
h = ax_dens.hist2d(embedding[:, 0], embedding[:, 1],
                   bins=60, cmap="magma",
                   norm=plt.matplotlib.colors.LogNorm())
plt.colorbar(h[3], ax=ax_dens, label="log count", fraction=0.02, pad=0.01)
ax_dens.set_xlabel("t-SNE 1"); ax_dens.set_ylabel("t-SNE 2")
ax_dens.set_title("Point Density (log scale)", fontsize=10)
plt.tight_layout(); st.pyplot(fig_dens); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# VIZ 3 — Marginal distributions (t-SNE 1 & 2)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="ct">Marginal Distributions of t-SNE Dimensions</div>',
            unsafe_allow_html=True)

fig_marg, axes = plt.subplots(1, 2, figsize=(12, 3.2), facecolor="#080b12")
for ax in axes: ax.set_facecolor("#0c0f1a")

for idx, (ax, label, col) in enumerate(zip(
        axes,
        ["t-SNE 1", "t-SNE 2"],
        [embedding[:, 0], embedding[:, 1]])):
    ax.hist(col, bins=60, color=PALETTE[idx], alpha=0.80, edgecolor="none")
    ax.set_xlabel(label, color="#4a5580", fontsize=8)
    ax.set_ylabel("Count", color="#4a5580", fontsize=8)
    ax.tick_params(colors="#4a5580", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2040")
    ax.set_title(f"Distribution of {label}", color="#c0c8e0", fontsize=9)

plt.tight_layout(); st.pyplot(fig_marg); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# VIZ 4 — Feature correlation with t-SNE axes (Pearson r)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="ct">Feature Correlation with t-SNE Axes</div>',
            unsafe_allow_html=True)
st.caption("Pearson r between each original feature and each t-SNE dimension. "
           "High |r| hints at which features drive the embedding geometry.")

corr_data = {}
for i in range(min(n_ss, 3)):
    corr_data[f"TSNE{i+1}"] = [
        np.corrcoef(X_ss[:, j], embedding[:, i])[0, 1]
        for j in range(len(feature_cols))
    ]
df_corr = pd.DataFrame(corr_data, index=feature_cols)

fig_corr, ax_corr = plt.subplots(figsize=(12, max(2.5, len(feature_cols) * 0.45 + 1.2)),
                                  facecolor="#080b12")
ax_corr.set_facecolor("#080b12")
im = ax_corr.imshow(df_corr.values.T, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax_corr, fraction=0.02, pad=0.01, label="Pearson r")
ax_corr.set_xticks(range(len(feature_cols)))
ax_corr.set_xticklabels(feature_cols, rotation=40, ha="right",
                         fontsize=8, color="#c0c8e0")
ax_corr.set_yticks(range(len(df_corr.columns)))
ax_corr.set_yticklabels(df_corr.columns, fontsize=8, color="#c0c8e0")
ax_corr.set_title("Feature ↔ t-SNE Correlation", color="#c0c8e0", fontsize=10)
for sp in ax_corr.spines.values(): sp.set_edgecolor("#1a2040")
ax_corr.tick_params(colors="#4a5580")
for i, ycol in enumerate(df_corr.columns):
    for j in range(len(feature_cols)):
        ax_corr.text(j, i, f"{df_corr.values[j, i]:.2f}",
                     ha="center", va="center",
                     fontsize=6.5, color="#dde1f0", fontfamily="monospace")
plt.tight_layout(); st.pyplot(fig_corr); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# VIZ 5 — Interactive 3-D (only when n_components == 3)
# ══════════════════════════════════════════════════════════════════════════════
if n_ss == 3:
    st.markdown('<div class="ct">Interactive 3-D t-SNE Scatter — rotate me!</div>',
                unsafe_allow_html=True)
    st.caption("Drag to rotate · Scroll to zoom · Double-click to reset")

    marker_color = color_vals if color_vals is not None else PALETTE[0]
    colorscale   = "Plasma" if color_vals is not None else None

    fig_3d = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
        mode="markers",
        marker=dict(
            size=3,
            color=marker_color,
            colorscale=colorscale if colorscale else [[0, PALETTE[0]], [1, PALETTE[0]]],
            opacity=0.72,
            line=dict(width=0),
            colorbar=dict(title=color_label, thickness=10,
                          tickfont=dict(color="#4a5580", size=9))
            if color_vals is not None else dict()
        ),
        text=[f"TSNE1:{embedding[i,0]:.2f}  TSNE2:{embedding[i,1]:.2f}  TSNE3:{embedding[i,2]:.2f}"
              for i in range(len(embedding))],
        hoverinfo="text", name="Samples"
    )])

    fig_3d.update_layout(
        paper_bgcolor="#080b12", plot_bgcolor="#0c0f1a",
        font=dict(family="JetBrains Mono", color="#8892b0", size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        height=580,
        title=dict(text=f"3-D t-SNE Embedding  (perplexity={p['perplexity']}  ·  KL={kl_str})",
                   font=dict(color="#8892b0", size=13)),
        scene=dict(
            bgcolor="#0c0f1a",
            xaxis=dict(backgroundcolor="#0c0f1a", gridcolor="#1a2040",
                       showbackground=True, tickfont=dict(color="#4a5580"),
                       title="t-SNE 1"),
            yaxis=dict(backgroundcolor="#0c0f1a", gridcolor="#1a2040",
                       showbackground=True, tickfont=dict(color="#4a5580"),
                       title="t-SNE 2"),
            zaxis=dict(backgroundcolor="#0c0f1a", gridcolor="#1a2040",
                       showbackground=True, tickfont=dict(color="#4a5580"),
                       title="t-SNE 3"),
        ),
        legend=dict(bgcolor="#0c0f1a", bordercolor="#1a2040", borderwidth=1,
                    font=dict(color="#8892b0", size=10)),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### 🏷️ Transformed Data")
st.markdown('<div class="ct">Original dataset with t-SNE coordinates appended</div>',
            unsafe_allow_html=True)
st.dataframe(df_result, use_container_width=True, height=320)

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER SENSITIVITY GUIDE
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📖 t-SNE Parameter Guide", expanded=False):
    st.markdown("""
| Parameter | Typical range | Effect |
|---|---|---|
| **Perplexity** | 5 – 50 | Controls neighbourhood size. Low → tight local clusters. High → more global structure preserved. |
| **Learning rate** | 10 – 1000 | Too low → dense ball. Too high → uniform spacing. 200 is a safe default. |
| **Iterations** | 500 – 2000 | More iterations → finer cluster separation, slower. |
| **Early exaggeration** | 4 – 20 | Amplifies inter-cluster gaps in early phase. Higher → more separated clusters. |
| **Init** | pca / random | PCA init is faster, more reproducible, and generally preferred. |
""")
    st.markdown("""
**Interpreting the embedding:**
- Cluster *shape* and *size* are not meaningful — only membership is.
- Distance *between* clusters is not meaningful — only relative proximity within a cluster is.
- Re-running may produce mirrored/rotated results; this is normal.
- If clusters merge or explode, try adjusting perplexity and learning rate.
""")

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOADS + SAVED BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.download_button("⬇ Download Transformed CSV",
                       df_result.to_csv(index=False).encode(),
                       file_name=f"tsne_n{n_ss}_p{p['perplexity']}.csv",
                       mime="text/csv", use_container_width=True)
with c2:
    with open(model_path, "rb") as f:
        st.download_button("⬇ Download Embedding (.pkl)", f.read(),
                           file_name=os.path.basename(model_path),
                           mime="application/octet-stream",
                           use_container_width=True)

abs_model = os.path.abspath(model_path)
abs_csv   = os.path.abspath(out_csv)
ts        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""
<div class="saved-banner">
  ✔ Embedding saved to &nbsp;<strong>{abs_model}</strong><br>
  ✔ Transformed data saved to &nbsp;<strong>{abs_csv}</strong><br>
  <span>Saved at {ts} &nbsp;·&nbsp; n_components={n_ss}
  &nbsp;·&nbsp; perplexity={p['perplexity']}
  &nbsp;·&nbsp; learning_rate={p['learning_rate']}
  &nbsp;·&nbsp; iterations={p['n_iter']}
  &nbsp;·&nbsp; KL divergence={kl_str}</span>
</div>
""", unsafe_allow_html=True)