#!/usr/bin/env python3
"""
Regresi Berganda & PLS - Insurance Dataset
Website interaktif dalam satu file Python (Flask)
Variabel X: age, bmi, children  |  Variabel Y: charges
Lengkap dengan OLS Diagnostics & PLS (Scores + Loadings)
"""

import io, base64, json, os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from flask import Flask, render_template_string, jsonify, request
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Load & proses data ────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "insurance.csv")
df_raw = pd.read_csv(CSV_PATH)
df = df_raw.copy()
df["smoker_num"] = (df["smoker"] == "yes").astype(int)
df["sex_num"]    = (df["sex"] == "male").astype(int)

X_COLS = ["age", "bmi", "children"]
Y_COL  = "charges"

X = df[X_COLS].values
y = df[Y_COL].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Model OLS (Global) ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_global = model.predict(X_test)

r2_global   = r2_score(y_test, y_pred_global)
rmse_global = np.sqrt(mean_squared_error(y_test, y_pred_global))
mae_global  = mean_absolute_error(y_test, y_pred_global)
n_g = len(y_test); p_g = len(X_COLS)
r2_adj_global = 1 - (1 - r2_global) * (n_g - 1) / (n_g - p_g - 1)


# ── Helpers Tema ──────────────────────────────────────────────────────────────
def get_theme_colors(is_dark):
    if is_dark:
        return "#1E1E1E", "#F3F4F6", "#9CA3AF" # BG_CARD, TEXT, TEXT_MUT (Dark)
    return "#ffffff", "#1C1C1C", "#6B7280"     # BG_CARD, TEXT, TEXT_MUT (Light)

def fig_to_b64(fig, bg_color):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=bg_color, edgecolor="none")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data


# ── Plot 1: Heatmap ───────────────────────────────────────────────────────────
def make_heatmap(df_sub, is_dark=False):
    bg_color, text_color, mut_color = get_theme_colors(is_dark)
    corr = df_sub[X_COLS + [Y_COL]].corr()
    fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    cmap = sns.diverging_palette(250, 20, as_cmap=True) 
    annot_kws_color = "#ffffff" if is_dark else "#1C1C1C"
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.8, linecolor=bg_color, ax=ax,
                annot_kws={"size": 9, "weight": "bold", "color": annot_kws_color},
                cbar_kws={"shrink": 0.7}, vmin=-1, vmax=1)
                
    ax.set_title(f"Heatmap Korelasi  (n={len(df_sub)})", color=text_color,
                 fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors=mut_color, labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", color=mut_color)
    plt.setp(ax.get_yticklabels(), rotation=0, color=mut_color)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=mut_color, labelsize=7.5)
    fig.tight_layout(pad=1.0)
    return fig_to_b64(fig, bg_color)


# ── Plot 2: PLS Scores ────────────────────────────────────────────────────────
def make_pls_scores(df_sub, is_dark=False):
    bg_color, text_color, mut_color = get_theme_colors(is_dark)
    LINE_COLOR = "#4B5563" if is_dark else "#e2e8f0"
    if len(df_sub) < 5: return fig_to_b64(plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)[0], bg_color)

    X_s = scaler.transform(df_sub[X_COLS].values)
    y_s = df_sub[Y_COL].values
    pls = PLSRegression(n_components=2)
    pls.fit(X_s, y_s)
    X_scores, _ = pls.transform(X_s, y_s)

    fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.scatter(X_scores[:, 0], X_scores[:, 1], alpha=0.6, s=16, color="#FF671D", linewidths=0)
    
    ax.set_title(f"PLS Scores Plot  (n={len(df_sub)})", color=text_color, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("PLS Component 1", color=mut_color, fontsize=9)
    ax.set_ylabel("PLS Component 2", color=mut_color, fontsize=9)
    ax.tick_params(colors=mut_color, labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor(LINE_COLOR)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig, bg_color)


# ── Plot 3: PLS Loadings (BARU) ───────────────────────────────────────────────
def make_pls_loadings(df_sub, is_dark=False):
    bg_color, text_color, mut_color = get_theme_colors(is_dark)
    LINE_COLOR = "#4B5563" if is_dark else "#e2e8f0"
    if len(df_sub) < 5: return fig_to_b64(plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)[0], bg_color)

    X_s = scaler.transform(df_sub[X_COLS].values)
    y_s = df_sub[Y_COL].values
    pls = PLSRegression(n_components=2)
    pls.fit(X_s, y_s)
    
    x_loadings = pls.x_loadings_

    fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    # Buat garis sumbu 0
    ax.axhline(0, color=LINE_COLOR, linestyle='--', linewidth=1)
    ax.axvline(0, color=LINE_COLOR, linestyle='--', linewidth=1)

    # Plot arah panah variabel
    for i, col in enumerate(X_COLS):
        ax.arrow(0, 0, x_loadings[i, 0], x_loadings[i, 1], color="#22C55E", 
                 alpha=0.8, head_width=0.03, head_length=0.03, linewidth=1.5)
        ax.text(x_loadings[i, 0] * 1.15, x_loadings[i, 1] * 1.15, col.upper(),
                color=text_color, ha='center', va='center', fontweight='bold', fontsize=9)
    
    max_val = np.max(np.abs(x_loadings)) * 1.4
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    ax.set_title(f"PLS Loadings Plot", color=text_color, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("PLS Component 1", color=mut_color, fontsize=9)
    ax.set_ylabel("PLS Component 2", color=mut_color, fontsize=9)
    ax.tick_params(colors=mut_color, labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor(LINE_COLOR)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig, bg_color)


# ── Plot 4: Scatter ───────────────────────────────────────────────────────────
def make_scatter(df_sub, is_dark=False):
    bg_color, text_color, mut_color = get_theme_colors(is_dark)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=bg_color)
    LINE_COLOR = "#4B5563" if is_dark else "#cbd5e1"
    colors = ["#FF671D", "#9CA3AF" if is_dark else "#0A0A0A", "#22C55E"]
    labels = ["Age (Usia)", "BMI", "Children"]

    for i, (col, clr, lbl) in enumerate(zip(X_COLS, colors, labels)):
        ax = axes[i]
        ax.set_facecolor(bg_color)
        x_vals = df_sub[col].values
        y_vals = df_sub[Y_COL].values
        ax.scatter(x_vals, y_vals, alpha=0.5, s=16, color=clr, linewidths=0)
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        ax.plot(x_line, m * x_line + b, color=LINE_COLOR, linewidth=1.8, alpha=0.9, linestyle="--")
        r_val = np.corrcoef(x_vals, y_vals)[0, 1]
        ax.set_title(f"{lbl}  (r={r_val:.3f})", color=text_color, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel(col, color=mut_color, fontsize=9)
        ax.set_ylabel("Charges (USD)" if i == 0 else "", color=mut_color, fontsize=9)
        ax.tick_params(colors=mut_color, labelsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
        for spine in ax.spines.values(): spine.set_edgecolor(LINE_COLOR)

    fig.suptitle(f"Scatter Plot: X vs Charges  (n={len(df_sub)})", color=text_color, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(pad=1.4)
    return fig_to_b64(fig, bg_color)


# ── Plot 5: Actual vs Predicted ───────────────────────────────────────────────
def make_actual_vs_pred(df_sub, is_dark=False):
    bg_color, text_color, mut_color = get_theme_colors(is_dark)
    LINE_COLOR = "#4B5563" if is_dark else "#e2e8f0"
    if len(df_sub) < 5: return fig_to_b64(plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)[0], bg_color)

    X_s = scaler.transform(df_sub[X_COLS].values)
    y_s = df_sub[Y_COL].values
    n_sub = len(df_sub)
    test_size = max(0.2, 5 / n_sub) if n_sub >= 10 else 0.5
    Xtr, Xte, ytr, yte = train_test_split(X_s, y_s, test_size=test_size, random_state=42)
    m_sub = LinearRegression().fit(Xtr, ytr)
    ypred = m_sub.predict(Xte)

    fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.scatter(yte, ypred, alpha=0.6, s=16, color="#FF671D", linewidths=0)
    
    lim = [min(yte.min(), ypred.min()) - 1000, max(yte.max(), ypred.max()) + 1000]
    PERFECT_LINE = "#9CA3AF" if is_dark else "#0A0A0A"
    ax.plot(lim, lim, color=PERFECT_LINE, linewidth=1.5, linestyle="--", label="Perfect fit")
    ax.set_xlim(lim); ax.set_ylim(lim)
    r2_s = r2_score(yte, ypred)
    
    ax.set_title(f"Actual vs Predicted  (R²={r2_s:.3f})", color=text_color, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Actual Charges (USD)", color=mut_color, fontsize=9)
    ax.set_ylabel("Predicted Charges (USD)", color=mut_color, fontsize=9)
    ax.tick_params(colors=mut_color, labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
    ax.legend(fontsize=8, labelcolor=mut_color, frameon=False)
    for spine in ax.spines.values(): spine.set_edgecolor(LINE_COLOR)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig, bg_color)


# ── Plot 6: Residual ──────────────────────────────────────────────────────────
def make_residual(df_sub, is_dark=False):
    bg_color, text_color, mut_color = get_theme_colors(is_dark)
    LINE_COLOR = "#4B5563" if is_dark else "#e2e8f0"
    if len(df_sub) < 5: return fig_to_b64(plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)[0], bg_color)

    X_s = scaler.transform(df_sub[X_COLS].values)
    y_s = df_sub[Y_COL].values
    n_sub = len(df_sub)
    test_size = max(0.2, 5 / n_sub) if n_sub >= 10 else 0.5
    Xtr, Xte, ytr, yte = train_test_split(X_s, y_s, test_size=test_size, random_state=42)
    m_sub = LinearRegression().fit(Xtr, ytr)
    ypred = m_sub.predict(Xte)
    residuals = yte - ypred

    fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    SCATTER_COLOR = "#9CA3AF" if is_dark else "#0A0A0A"
    ax.scatter(ypred, residuals, alpha=0.6, s=16, color=SCATTER_COLOR, linewidths=0)
    ax.axhline(0, color="#FF671D", linewidth=1.5, linestyle="--")
    
    ax.set_title(f"Residual Plot  (n={len(df_sub)})", color=text_color, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Predicted Charges (USD)", color=mut_color, fontsize=9)
    ax.set_ylabel("Residuals", color=mut_color, fontsize=9)
    ax.tick_params(colors=mut_color, labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
    for spine in ax.spines.values(): spine.set_edgecolor(LINE_COLOR)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig, bg_color)


# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Regresi Berganda & PLS — Insurance Dataset</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@1,500;1,600;1,700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #FDF7EC; 
  --card: #FFFFFF;
  --nav-bg: rgba(15, 15, 15, 0.65); 
  --nav-border: rgba(255, 255, 255, 0.1);
  --nav-text: #FFFFFF;
  --accent: #FF671D; 
  --text: #1C1C1C;
  --muted: #6B7280;
  --border: #EAE0D3;
  --shadow: rgba(0,0,0,0.04);
  --pill-bg: #F1F5F9;
  --table-hover: #F8FAFC;
}
[data-theme="dark"] {
  --bg: #121212;
  --card: #1E1E1E;
  --nav-bg: rgba(25, 25, 25, 0.7); 
  --nav-border: rgba(255, 255, 255, 0.05);
  --nav-text: #F3F4F6;
  --accent: #FF671D;
  --text: #F3F4F6;
  --muted: #9CA3AF;
  --border: #374151;
  --shadow: rgba(0,0,0,0.3);
  --pill-bg: #374151;
  --table-hover: #292929;
}
*{box-sizing:border-box;margin:0;padding:0}
html { scroll-behavior: smooth; scroll-padding-top: 100px; }
body{background:var(--bg);color:var(--text);font-family:"Plus Jakarta Sans",sans-serif;min-height:100vh; transition: background 0.3s, color 0.3s;}

.nav-wrapper { position: sticky; top: 1.5rem; z-index: 100; padding: 0 2rem; }
nav { background: var(--nav-bg); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid var(--nav-border); color: var(--nav-text); max-width: 1200px; margin: 0 auto; border-radius: 99px; padding: 0.8rem 1.5rem 0.8rem 2rem; display: flex; align-items: center; box-shadow: 0 10px 30px rgba(0,0,0,0.15); transition: all 0.3s ease; }
.nav-logo { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; }
.nav-logo span { font-weight: 400; }
.nav-links { display: flex; gap: 2rem; margin-left: auto; margin-right: 2rem; align-items: center; }
.nav-links a { font-size: 0.9rem; font-weight: 500; color: #E5E5E5; text-decoration: none; transition: color 0.2s; }
.nav-links a:hover { color: var(--accent); }
.nav-actions { display: flex; align-items: center; gap: 0.8rem; }
.nav-btn { background: linear-gradient(135deg, #FF7B00, #F95000); color: white; border: none; border-radius: 99px; padding: 0.5rem 1.2rem; font-family: inherit; font-weight: 700; font-size: 0.85rem; cursor: pointer; }
.theme-toggle { background: transparent; border: 1px solid rgba(255,255,255,0.2); color: white; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 1rem; transition: background 0.2s; }
.theme-toggle:hover { background: rgba(255,255,255,0.1); }

.hero{text-align: center; padding: 5rem 2rem 3rem; max-width:900px; margin:0 auto}
.hero h1{font-size:clamp(2.5rem, 5vw, 4.2rem); font-weight:800; letter-spacing:-0.03em; line-height:1.15; color:var(--text);}
.hero h1 .serif { font-family: 'Playfair Display', serif; font-style: italic; font-weight: 600; color: var(--accent); }
.hero p{margin-top:1.5rem; color:var(--muted); font-size:1.1rem; line-height:1.6; font-weight: 500;}
.badge-row{display:flex;flex-wrap:wrap;justify-content: center;gap:0.6rem;margin-top:2rem}
.badge{background:transparent;border:1.5px solid var(--border);border-radius:99px; padding:0.4rem 1rem;font-size:0.85rem;color:var(--text);font-weight:600;}

.wrap{max-width:1200px;margin:0 auto;padding:0 2rem 5rem}
.section-title{font-size:1.5rem;font-weight:700;margin-bottom:1.2rem;color:var(--text); display:flex;align-items:center;gap:0.6rem; letter-spacing: -0.02em; margin-top:1.5rem;}
.section-title i { font-family: 'Playfair Display', serif; font-style: italic; color: var(--accent); }

.filter-bar{background:var(--card);border:none;border-radius:100px; padding:0.8rem 1.5rem;margin-bottom:2.5rem;display:flex;flex-wrap:wrap; align-items:center;gap:1rem;box-shadow:0 8px 30px var(--shadow)}
.filter-bar label{font-size:0.9rem;color:var(--text);font-weight:600}
.sel, .search-box{ background:var(--bg);border:1px solid var(--border);border-radius:99px; color:var(--text);padding:0.6rem 1.2rem;font-size:0.9rem;cursor:pointer;outline:none; font-family:inherit; transition: all 0.2s; font-weight: 500; }
.sel:focus, .search-box:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(255,103,29,0.15)}
.search-box{width:220px}
.search-box::placeholder{color:var(--muted)}
.filter-info{margin-left:auto;font-size:0.9rem;color:var(--muted); font-weight: 500;}
.filter-info strong{color:var(--text)}
.btn-refresh{background:var(--text);border:none;border-radius:99px;color:var(--bg); padding:0.6rem 1.5rem;font-size:0.9rem;font-weight:600;cursor:pointer; transition:all 0.2s;white-space:nowrap; font-family: inherit;}
.btn-refresh:hover{opacity: 0.8;}

.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1.5rem;margin-bottom:3.5rem}
.metric{background:var(--card);border:none;border-radius:24px;padding:1.8rem; box-shadow:0 8px 30px var(--shadow); text-align: center;}
.metric-label{font-size:0.8rem;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:0.05em}
.metric-value{font-size:2.2rem;font-weight:800;margin:0.5rem 0 0.2rem;color:var(--text); font-family:"JetBrains Mono",monospace; letter-spacing: -1px;}
.metric-desc{font-size:0.85rem;color:var(--muted); font-weight: 500;}

.eq-card{background:var(--card);border:none;border-radius:24px; padding:2rem;margin-bottom:3.5rem;overflow-x:auto;box-shadow:0 8px 30px var(--shadow)}
.eq-title{font-size:0.85rem;color:var(--accent);font-weight:700;text-transform:uppercase;letter-spacing: 0.05em; margin-bottom:1rem}
.eq-formula{font-family:"JetBrains Mono",monospace;font-size:clamp(1rem,2vw,1.15rem); color:var(--text);line-height:1.8;background:var(--bg);padding:1.5rem;border-radius:16px; border:1px solid var(--border); font-weight: 500;}
.eq-formula .var{color:var(--text);font-weight:700}
.eq-formula .coef{color:var(--accent)}

.charts-2{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:3.5rem}
.charts-1{margin-bottom:3.5rem}
.chart-card{background:var(--card);border:none;border-radius:24px;overflow:hidden; box-shadow:0 8px 30px var(--shadow); padding: 1rem; transition: background 0.3s;}
.chart-card img{width:100%;border-radius: 12px;display:block;}
.chart-card .chart-label{padding:1rem 1rem 0.5rem;font-size:0.9rem;color:var(--muted); text-align:center; font-weight: 500;}

.tbl-wrap{background:var(--card);border:none;border-radius:24px;overflow:hidden; box-shadow:0 8px 30px var(--shadow); padding: 1rem;}
.tbl-scroll{overflow-x:auto;max-height:500px;overflow-y:auto; border-radius: 12px;}
.tbl-scroll::-webkit-scrollbar{width:8px;height:8px}
.tbl-scroll::-webkit-scrollbar-track{background:var(--card)}
.tbl-scroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
table{width:100%;border-collapse:collapse;font-size:0.9rem}
thead{position:sticky;top:0;z-index:5}
thead th{background:var(--bg);color:var(--text);font-weight:700; padding:1rem 1.2rem;text-align:left;white-space:nowrap; border-bottom:2px solid var(--border)}
tbody tr{border-bottom:1px solid var(--border);transition:background 0.2s}
tbody tr:hover{background:var(--table-hover)}
tbody td{padding:0.9rem 1.2rem;color:var(--text);white-space:nowrap; font-weight: 500;}
tbody td.num{font-family:"JetBrains Mono",monospace;color:var(--muted)}
tbody td.charge{color:var(--text);font-weight:700;font-family:"JetBrains Mono",monospace}
.pill{display:inline-block;border-radius:99px;padding:0.25rem 0.8rem;font-size:0.8rem;font-weight:700}
.pill-yes{background:rgba(255,103,29,0.15);color:var(--accent);}
.pill-no{background:var(--pill-bg);color:var(--muted);}
.pill-m{background:rgba(34,197,94,0.15);color:#22C55E;}
.pill-f{background:rgba(217,119,6,0.15);color:#D97706;}
.tbl-footer{padding:1.2rem 1rem 0.5rem;font-size:0.9rem;color:var(--muted); font-weight: 600; display:flex;justify-content:space-between;align-items:center}

.predict-card{background:var(--card);border:none;border-radius:24px; padding:2.5rem;margin-bottom:3.5rem;box-shadow:0 8px 30px var(--shadow)}
.input-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1.5rem;margin-bottom:2rem}
.inp-group label{display:block;font-size:0.9rem;color:var(--text);margin-bottom:0.6rem;font-weight:700}
.inp-group input{width:100%;background:var(--bg);border:1px solid var(--border); border-radius:12px;color:var(--text);padding:0.8rem 1rem;font-size:1rem; outline:none;transition:all 0.2s;font-family:"JetBrains Mono",monospace; font-weight: 600;}
.inp-group input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(255,103,29,0.15)}
.btn-orange{background:linear-gradient(135deg, #FF7B00, #F95000);border:none; border-radius:99px;color:#fff;padding:0.8rem 2rem;font-size:1rem;font-weight:700; cursor:pointer;transition:all 0.2s; font-family: inherit; box-shadow: 0 4px 15px rgba(249, 80, 0, 0.25);}
.btn-orange:hover{transform: translateY(-2px); box-shadow: 0 6px 20px rgba(249, 80, 0, 0.35);}
.pred-result{margin-top:2rem;padding:1.5rem 2rem;background:var(--bg); border-radius:16px;display:none; text-align: center; border: 1px solid var(--border);}
.pred-result .pred-label{font-size:0.9rem;color:var(--muted);font-weight:700;text-transform:uppercase; letter-spacing: 0.05em;}
.pred-result .pred-val{font-size:2.5rem;font-weight:800;color:var(--accent); font-family:"JetBrains Mono",monospace;margin-top:0.5rem; letter-spacing: -1px;}

.coef-table{width:100%;border-collapse:collapse;font-size:0.9rem;margin-top:2rem}
.coef-table th{background:transparent;color:var(--muted);font-size:0.8rem;font-weight:700; text-transform:uppercase;padding:1rem;text-align:left;border-bottom:2px solid var(--border)}
.coef-table td{padding:1rem;border-bottom:1px solid var(--border);font-family:"JetBrains Mono",monospace; font-weight: 500;}
.coef-table td:first-child{font-family:"Plus Jakarta Sans",sans-serif;font-weight:700;color:var(--text)}
.coef-bar{height:8px;border-radius:4px;background:var(--accent);margin-top:8px;transition:width 0.5s}

.chart-spinner{display:flex;align-items:center;justify-content:center; height:160px;color:var(--muted);font-size:0.9rem;gap:0.8rem; font-weight: 600;}
.spin{width:22px;height:22px;border:3px solid var(--border); border-top-color:var(--accent);border-radius:50%;animation:spin 0.8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

@media(max-width:800px){ .charts-2{grid-template-columns:1fr} .wrap{padding:0 1rem 3rem} .nav-wrapper { padding: 0 1rem; } .nav-links { display: none; } .hero h1 { font-size: 2.2rem; } .filter-bar { border-radius: 20px; } }
</style>
</head>
<body>

<div class="nav-wrapper">
  <nav>
    <div class="nav-logo">Regresi<span>Ganda</span></div>
    <div class="nav-links">
      <a href="#model">Model</a>
      <a href="#dataset">Dataset</a>
      <a href="#pls">Analisis PLS</a>
      <a href="#diagnostik">Diagnostik OLS</a>
    </div>
    <div class="nav-actions">
      <button class="nav-btn">n = {{ n_total }}</button>
      <button class="theme-toggle" id="theme-btn" title="Toggle Dark Mode">🌙</button>
    </div>
  </nav>
</div>

<div class="hero">
  <h1>Analisis <span class="serif">Regresi Berganda</span><br>&amp; <span class="serif">PLS</span> Insurance</h1>
  <p>Belajar memprediksi biaya asuransi (<strong>charges</strong>) berdasarkan usia, BMI, dan jumlah anak dengan pendekatan OLS (Regresi Linear) dan Regresi PLS, Data bersumber dari kaggle.<br><br><span style="color:var(--accent); font-weight:800; letter-spacing:1px; font-size:0.95rem;">— KELOMPOK 2 —</span></p>
  <div class="badge-row">
    <span class="badge">X₁: age</span>
    <span class="badge">X₂: bmi</span>
    <span class="badge">X₃: children</span>
    <span class="badge" style="color:var(--accent); border-color:var(--accent)">Y: charges</span>
  </div>
</div>

<div class="wrap">

  <div class="section-title" id="model">Metrik <i>Model OLS</i></div>
  <div class="metrics">
    <div class="metric">
      <div class="metric-label">R² Score</div>
      <div class="metric-value">{{ "%.4f"|format(r2) }}</div>
      <div class="metric-desc">Koefisien determinasi</div>
    </div>
    <div class="metric">
      <div class="metric-label">Adj. R²</div>
      <div class="metric-value">{{ "%.4f"|format(r2_adj) }}</div>
      <div class="metric-desc">Adjusted R-squared</div>
    </div>
    <div class="metric">
      <div class="metric-label">RMSE</div>
      <div class="metric-value" style="color:var(--accent)">${{ "{:,.0f}".format(rmse) }}</div>
      <div class="metric-desc">Root Mean Sq. Error</div>
    </div>
    <div class="metric">
      <div class="metric-label">MAE</div>
      <div class="metric-value">${{ "{:,.0f}".format(mae) }}</div>
      <div class="metric-desc">Mean Absolute Error</div>
    </div>
  </div>

  <div class="eq-card">
    <div class="eq-title">Model Regresi Linear: Y = b₀ + b₁X₁ + b₂X₂ + b₃X₃</div>
    <div class="eq-formula">
      <span class="var">charges</span> = 
      <span class="coef">{{ "{:,.2f}".format(intercept) }}</span> +
      (<span class="coef">{{ "{:,.2f}".format(coefs[0]) }}</span> × <span class="var">age</span>) +
      (<span class="coef">{{ "{:,.2f}".format(coefs[1]) }}</span> × <span class="var">bmi</span>) +
      (<span class="coef">{{ "{:,.2f}".format(coefs[2]) }}</span> × <span class="var">children</span>)
    </div>
    <table class="coef-table">
      <thead><tr><th>Variabel</th><th>Koefisien (Scaled)</th><th>Kontribusi Relatif</th></tr></thead>
      <tbody>
        {% for v, c, b in coef_rows %}
        <tr>
          <td>{{ v }}</td>
          <td>{{ "{:,.4f}".format(c) }}</td>
          <td>
            <div>{{ "%.1f"|format(b) }}%</div>
            <div class="coef-bar" style="width:{{ b }}%"></div>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="section-title" id="prediksi">Prediksi <i>Masa Depanmu</i></div>
  <div class="predict-card">
    <div class="input-grid">
      <div class="inp-group">
        <label>Age (Usia)</label>
        <input type="number" id="p-age" placeholder="e.g. 35" min="18" max="100" value="35"/>
      </div>
      <div class="inp-group">
        <label>BMI</label>
        <input type="number" id="p-bmi" placeholder="e.g. 27.5" step="0.1" min="10" max="60" value="27.5"/>
      </div>
      <div class="inp-group">
        <label>Children (Jumlah Anak)</label>
        <input type="number" id="p-children" placeholder="e.g. 2" min="0" max="10" value="2"/>
      </div>
    </div>
    <button class="btn-orange" onclick="predict()">Apply Prediction</button>
    <div class="pred-result" id="pred-result">
      <div class="pred-label">Estimasi Biaya Asuransi</div>
      <div class="pred-val" id="pred-val">–</div>
    </div>
  </div>

  <div class="section-title" id="dataset">Dataset <i>&amp; Filter</i></div>
  <div class="filter-bar">
    <label>Show:</label>
    <select class="sel" id="row-sel" onchange="applyFilters()">
      <option value="100">100 baris</option>
      <option value="200">200 baris</option>
      <option value="500">500 baris</option>
      <option value="all">Semua ({{ n_total }})</option>
    </select>

    <input class="search-box" id="search" placeholder="Cari data..." oninput="applyFilters()"/>

    <select class="sel" id="smoker-sel" onchange="applyFilters()">
      <option value="all">Smoker: All</option>
      <option value="yes">Smoker: Yes</option>
      <option value="no">Smoker: No</option>
    </select>

    <select class="sel" id="sort-col" onchange="applyFilters()">
      <option value="charges-desc">Sort: Charges ↓</option>
      <option value="charges-asc">Sort: Charges ↑</option>
      <option value="age-desc">Sort: Age ↓</option>
      <option value="age-asc">Sort: Age ↑</option>
      <option value="bmi-desc">Sort: BMI ↓</option>
    </select>

    <button class="btn-refresh" id="btn-update" onclick="updateCharts()">Sync Charts</button>
    <span class="filter-info">Aktif: <strong id="active-count">–</strong> baris</span>
  </div>

  <div class="tbl-wrap" style="margin-bottom:3.5rem">
    <div class="tbl-scroll">
      <table id="main-table">
        <thead>
          <tr>
            <th>#</th><th>Age</th><th>Sex</th><th>BMI</th>
            <th>Children</th><th>Smoker</th><th>Region</th><th>Charges (USD)</th>
          </tr>
        </thead>
        <tbody id="tbl-body"></tbody>
      </table>
    </div>
    <div class="tbl-footer">
      <span id="tbl-info">Memuat data…</span>
    </div>
  </div>

  <div class="section-title" id="pls">Eksplorasi Heatmap <i>&amp; Model PLS</i></div>
  <div class="charts-2">
    <div class="chart-card" id="card-heatmap">
      <div class="chart-spinner"><div class="spin"></div> Memproses...</div>
      <div class="chart-label" id="lbl-heatmap">1. Heatmap Korelasi Pearson</div>
    </div>
    <div class="chart-card" id="card-pls-scores">
      <div class="chart-spinner"><div class="spin"></div> Memproses...</div>
      <div class="chart-label" id="lbl-pls-scores">2. PLS Scores Plot (Persebaran Data)</div>
    </div>
  </div>
  
  <div class="charts-2">
    <div class="chart-card" id="card-pls-loadings">
      <div class="chart-spinner"><div class="spin"></div> Memproses...</div>
      <div class="chart-label" id="lbl-pls-loadings">3. PLS Loadings (Arah/Pengaruh Variabel X)</div>
    </div>
    <div class="chart-card" style="display:flex; align-items:center; justify-content:center; padding:2rem; text-align:center; background: transparent; box-shadow:none;">
      <p style="color:var(--muted); line-height:1.6; font-size:0.95rem;">
        <strong>Interpretasi PLS:</strong><br>
        <strong>Scores Plot</strong> melihat apakah ada pengelompokan (klaster) pada baris data berdasarkan komponen latent.<br><br>
        <strong>Loadings Plot</strong> menunjukkan seberapa besar kontribusi variabel <code style="color:var(--accent)">age, bmi, children</code> dalam memengaruhi model PLS. Panah yang panjang menjauhi pusat (0,0) memiliki bobot pengaruh yang paling kuat.
      </p>
    </div>
  </div>

  <div class="section-title" id="diagnostik">Diagnostik Regresi <i>(OLS Scatter &amp; Error)</i></div>
  <div class="charts-1">
    <div class="chart-card" id="card-scatter">
      <div class="chart-spinner"><div class="spin"></div> Memproses...</div>
      <div class="chart-label" id="lbl-scatter">4. Hubungan X vs Charges beserta Garis Regresi Linier</div>
    </div>
  </div>

  <div class="charts-2">
    <div class="chart-card" id="card-avp">
      <div class="chart-spinner"><div class="spin"></div> Memproses...</div>
      <div class="chart-label" id="lbl-avp">5. Actual vs Predicted (OLS)</div>
    </div>
    <div class="chart-card" id="card-res">
      <div class="chart-spinner"><div class="spin"></div> Memproses...</div>
      <div class="chart-label" id="lbl-res">6. Residual Plot (Homoskedastisitas)</div>
    </div>
  </div>

</div>

<script>
const RAW = {{ table_data | safe }};
let currentFiltered = [];

const themeBtn = document.getElementById("theme-btn");
const currentTheme = localStorage.getItem("theme") || "light";

if (currentTheme === "dark") {
  document.documentElement.setAttribute("data-theme", "dark");
  themeBtn.textContent = "☀️";
}

themeBtn.addEventListener("click", () => {
  let theme = document.documentElement.getAttribute("data-theme");
  if (theme === "dark") {
    document.documentElement.removeAttribute("data-theme");
    localStorage.setItem("theme", "light");
    themeBtn.textContent = "🌙";
  } else {
    document.documentElement.setAttribute("data-theme", "dark");
    localStorage.setItem("theme", "dark");
    themeBtn.textContent = "☀️";
  }
  updateCharts(); 
});


async function predict(){
  const age = document.getElementById("p-age").value;
  const bmi = document.getElementById("p-bmi").value;
  const children = document.getElementById("p-children").value;
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({age, bmi, children})
    });
    const data = await res.json();
    document.getElementById("pred-result").style.display = "block";
    document.getElementById("pred-val").textContent = 
      "$" + data.prediction.toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:2});
  } catch(e) { alert("Gagal memprediksi"); }
}

function applyFilters(){
  const limit    = document.getElementById("row-sel").value;
  const search   = document.getElementById("search").value.toLowerCase();
  const smoker   = document.getElementById("smoker-sel").value;
  const [sKey, sDir] = document.getElementById("sort-col").value.split("-");

  let data = RAW.filter(r => {
    if(smoker !== "all" && r.smoker !== smoker) return false;
    if(search){
      if(!Object.values(r).join(" ").toLowerCase().includes(search)) return false;
    }
    return true;
  });

  data.sort((a,b) => {
    const va = parseFloat(a[sKey]), vb = parseFloat(b[sKey]);
    return sDir === "asc" ? va-vb : vb-va;
  });

  const total = data.length;
  if(limit !== "all") data = data.slice(0, parseInt(limit));
  currentFiltered = data; 

  const tbody = document.getElementById("tbl-body");
  tbody.innerHTML = data.map((r,i) => `
    <tr>
      <td class="num">${i+1}</td>
      <td class="num">${r.age}</td>
      <td><span class="pill ${r.sex==='male'?'pill-m':'pill-f'}">${r.sex}</span></td>
      <td class="num">${parseFloat(r.bmi).toFixed(2)}</td>
      <td class="num">${r.children}</td>
      <td><span class="pill ${r.smoker==='yes'?'pill-yes':'pill-no'}">${r.smoker}</span></td>
      <td style="color:var(--muted)">${r.region}</td>
      <td class="charge">$${parseFloat(r.charges).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2})}</td>
    </tr>`).join("");

  document.getElementById("tbl-info").textContent = `Menampilkan ${data.length} dari ${total} baris`;
  document.getElementById("active-count").textContent = data.length;
}

function setCardImg(cardId, b64){
  const card = document.getElementById(cardId);
  const lbl  = card.querySelector(".chart-label");
  const old = card.querySelector(".chart-spinner, img");
  if(old) old.remove();
  const img = document.createElement("img");
  img.src = "data:image/png;base64," + b64;
  card.insertBefore(img, lbl);
}

function setSpinner(cardId){
  const card = document.getElementById(cardId);
  const lbl  = card.querySelector(".chart-label");
  const old  = card.querySelector("img");
  if(old) old.remove();
  if(!card.querySelector(".chart-spinner")){
    const s = document.createElement("div");
    s.className = "chart-spinner";
    s.innerHTML = '<div class="spin"></div> Memproses Visualisasi…';
    card.insertBefore(s, lbl);
  }
}

async function updateCharts(){
  const btn = document.getElementById("btn-update");
  btn.disabled = true;
  btn.textContent = "Syncing...";

  const rows = currentFiltered.map(r => ({
    age: parseFloat(r.age), bmi: parseFloat(r.bmi),
    children: parseFloat(r.children), charges: parseFloat(r.charges)
  }));
  
  const currentThemeStr = document.documentElement.getAttribute("data-theme") || "light";

  ["card-heatmap", "card-pls-scores", "card-pls-loadings", "card-scatter", "card-avp", "card-res"].forEach(setSpinner);

  try {
    const res = await fetch("/charts", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ rows: rows, theme: currentThemeStr })
    });
    const data = await res.json();
    if(data.error){ alert("Error: " + data.error); return; }

    setCardImg("card-heatmap", data.heatmap);
    setCardImg("card-pls-scores", data.pls_scores);
    setCardImg("card-pls-loadings", data.pls_loadings); // Load grafik PLS baru
    setCardImg("card-scatter", data.scatter);
    setCardImg("card-avp",     data.act_vs_pred);
    setCardImg("card-res",     data.residual);

  } catch(e){
    alert("Gagal memuat grafik: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Sync Charts";
  }
}

applyFilters(); 
updateCharts(); 
</script>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    coefs_raw    = model.coef_
    intercept_raw = model.intercept_
    coefs_orig   = coefs_raw / scaler.scale_
    intercept_orig = intercept_raw - np.dot(coefs_orig, scaler.mean_)

    abs_coefs = np.abs(coefs_raw)
    pct = abs_coefs / abs_coefs.sum() * 100
    coef_rows = list(zip(X_COLS, coefs_orig, pct))

    table_data = df_raw[["age","sex","bmi","children","smoker","region","charges"]].to_dict("records")

    return render_template_string(
        HTML,
        r2=r2_global, r2_adj=r2_adj_global, rmse=rmse_global, mae=mae_global,
        n_total=len(df_raw), n_train=len(X_train), n_test=len(X_test),
        intercept=intercept_orig,
        coefs=coefs_orig,
        coef_rows=coef_rows,
        table_data=json.dumps(table_data),
    )


@app.route("/charts", methods=["POST"])
def charts_route():
    try:
        body = request.get_json()
        rows = body.get("rows", [])
        is_dark = body.get("theme", "light") == "dark"

        if not rows:
            df_sub = df[X_COLS + [Y_COL]].copy()
        else:
            df_sub = pd.DataFrame(rows, columns=["age", "bmi", "children", "charges"])

        if len(df_sub) < 2:
            return jsonify({"error": "Data terlalu sedikit."}), 400

        return jsonify({
            "n":             len(df_sub),
            "heatmap":       make_heatmap(df_sub, is_dark),
            "pls_scores":    make_pls_scores(df_sub, is_dark),
            "pls_loadings":  make_pls_loadings(df_sub, is_dark), # Memanggil hasil render fungsi baru
            "scatter":       make_scatter(df_sub, is_dark),
            "act_vs_pred":   make_actual_vs_pred(df_sub, is_dark),
            "residual":      make_residual(df_sub, is_dark),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_route():
    body     = request.get_json()
    age      = float(body.get("age",      30))
    bmi      = float(body.get("bmi",      25))
    children = float(body.get("children", 0))
    X_in = scaler.transform([[age, bmi, children]])
    pred = model.predict(X_in)[0]
    return jsonify({"prediction": round(pred, 2)})


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Regresi Berganda & PLS — Insurance Dataset")
    print("  Buka browser: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)