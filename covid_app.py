import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 · Análisis de Evolución",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Space Mono', monospace; }
  .main { background: #04040f; color: #c8c8e8; }

  /* Header */
  .hero-block {
    background: linear-gradient(135deg, #04040f 0%, #0d0d22 100%);
    border: 1px solid rgba(0,245,212,0.15);
    border-left: 4px solid #00f5d4;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    border-radius: 0;
  }
  .hero-block h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    color: #eeeeff; margin: 0 0 0.3rem;
    letter-spacing: -0.02em;
  }
  .hero-block p { color: #6464a0; font-size: 0.82rem; margin: 0; }

  /* KPI cards */
  .kpi-grid { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
  .kpi-card {
    flex: 1; min-width: 160px;
    background: #080818;
    border: 1px solid rgba(0,245,212,0.12);
    border-top: 2px solid;
    padding: 1.2rem 1.5rem;
  }
  .kpi-card.cyan  { border-top-color: #00f5d4; }
  .kpi-card.purple{ border-top-color: #8b5cf6; }
  .kpi-card.pink  { border-top-color: #f0abfc; }
  .kpi-card.orange{ border-top-color: #fb923c; }
  .kpi-label { font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase; color: #6464a0; margin-bottom: 0.4rem; }
  .kpi-value { font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 800; color: #eeeeff; }
  .kpi-value.cyan  { color: #00f5d4; }
  .kpi-value.purple{ color: #8b5cf6; }
  .kpi-value.pink  { color: #f0abfc; }
  .kpi-value.orange{ color: #fb923c; }

  /* Section headers */
  .section-hdr {
    font-size: 0.68rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: #00f5d4;
    margin: 0 0 0.3rem;
  }
  .section-title {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 1.4rem; color: #eeeeff;
    letter-spacing: -0.02em; margin-bottom: 1.2rem;
  }

  /* Metric badge */
  .badge {
    display: inline-block;
    background: rgba(0,245,212,0.08);
    border: 1px solid rgba(0,245,212,0.3);
    color: #00f5d4;
    font-size: 0.72rem; padding: 0.3rem 0.8rem;
    margin-right: 0.5rem; margin-top: 0.5rem;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #080818 !important;
    border-right: 1px solid rgba(0,245,212,0.1);
  }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] .stMultiSelect label { color: #6464a0 !important; font-size: 0.72rem !important; }

  /* Divider */
  hr { border-color: rgba(0,245,212,0.1) !important; }

  /* Hide streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    cols = ["location", "date", "total_cases", "total_deaths",
            "total_tests", "new_cases", "new_deaths", "population",
            "continent", "total_vaccinations"]
    df = pd.read_csv(url, usecols=cols, parse_dates=["date"], low_memory=False)
    df.rename(columns={"total_cases": "cases", "total_deaths": "deaths"}, inplace=True)
    df = df[~df["location"].str.startswith("OWID_")]   # remove aggregate rows
    df = df[df["continent"].notna()]                    # only countries
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🦠 COVID-19 · Controles")
    st.markdown("---")

    with st.spinner("Cargando datos globales…"):
        df = load_data()

    countries = sorted(df["location"].unique())
    default_countries = ["Peru", "Colombia", "Brazil", "United States", "Spain"]
    default_countries = [c for c in default_countries if c in countries]

    st.markdown("**Comparación de países**")
    selected_countries = st.multiselect(
        "Selecciona países", countries,
        default=default_countries
    )

    st.markdown("---")
    st.markdown("**Forecasting**")
    forecast_country = st.selectbox(
        "País para forecast",
        countries,
        index=countries.index("Peru") if "Peru" in countries else 0
    )
    forecast_steps = st.slider("Días a predecir", 15, 90, 30, 5)
    train_pct = st.slider("% datos para entrenamiento", 70, 95, 85, 5)

    st.markdown("---")
    st.markdown("**Métrica a visualizar**")
    metric = st.radio(
        "Variable principal",
        ["cases", "deaths", "total_vaccinations"],
        format_func=lambda x: {
            "cases": "Casos confirmados",
            "deaths": "Muertes",
            "total_vaccinations": "Vacunaciones"
        }[x]
    )

    st.markdown("---")
    st.caption("Fuente: Our World in Data · OWID COVID-19 Dataset")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
  <h1>🦠 Análisis de la Evolución del COVID-19</h1>
  <p>Series temporales globales · Comparación de países · Forecasting con Suavizado Exponencial</p>
</div>
""", unsafe_allow_html=True)

# ── Global KPIs ───────────────────────────────────────────────────────────────
latest = df.sort_values("date").groupby("location").last().reset_index()
total_cases  = latest["cases"].sum()
total_deaths = latest["deaths"].sum()
total_vacc   = latest["total_vaccinations"].sum()
cfr = (total_deaths / total_cases * 100) if total_cases > 0 else 0

def fmt(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.1f}M"
    if n >= 1e3: return f"{n/1e3:.0f}K"
    return str(int(n))

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card cyan">
    <div class="kpi-label">Casos Confirmados</div>
    <div class="kpi-value cyan">{fmt(total_cases)}</div>
  </div>
  <div class="kpi-card purple">
    <div class="kpi-label">Muertes Totales</div>
    <div class="kpi-value purple">{fmt(total_deaths)}</div>
  </div>
  <div class="kpi-card pink">
    <div class="kpi-label">Vacunaciones</div>
    <div class="kpi-value pink">{fmt(total_vacc)}</div>
  </div>
  <div class="kpi-card orange">
    <div class="kpi-label">Tasa Letalidad (CFR)</div>
    <div class="kpi-value orange">{cfr:.2f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Section 1: Evolución Global ───────────────────────────────────────────────
st.markdown('<div class="section-hdr">// 01 — Evolución Global</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Casos · Muertes · Tests acumulados</div>', unsafe_allow_html=True)

global_daily = (
    df.groupby("date")[["cases", "deaths", "total_tests"]]
    .sum()
    .reset_index()
    .sort_values("date")
)

fig_global = make_subplots(specs=[[{"secondary_y": True}]])
fig_global.add_trace(go.Scatter(
    x=global_daily["date"], y=global_daily["cases"],
    name="Casos confirmados", line=dict(color="#00f5d4", width=2),
    fill="tozeroy", fillcolor="rgba(0,245,212,0.05)"
), secondary_y=False)
fig_global.add_trace(go.Scatter(
    x=global_daily["date"], y=global_daily["deaths"],
    name="Muertes", line=dict(color="#f0abfc", width=2)
), secondary_y=True)
fig_global.add_trace(go.Scatter(
    x=global_daily["date"], y=global_daily["total_tests"],
    name="Tests", line=dict(color="#8b5cf6", width=1.5, dash="dot")
), secondary_y=False)

fig_global.update_layout(
    paper_bgcolor="#04040f", plot_bgcolor="#080818",
    font=dict(family="Space Mono", color="#c8c8e8", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,245,212,0.2)", borderwidth=1),
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", showgrid=True),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", showgrid=True),
    yaxis2=dict(gridcolor="rgba(0,0,0,0)"),
    height=380
)
st.plotly_chart(fig_global, use_container_width=True)
st.markdown("---")

# ── Section 2: Comparación de Países ─────────────────────────────────────────
st.markdown('<div class="section-hdr">// 02 — Comparación de Países</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-title">Evolución de <span style="color:#00f5d4">'
            f'{"casos" if metric=="cases" else "muertes" if metric=="deaths" else "vacunaciones"}'
            f'</span> por país</div>', unsafe_allow_html=True)

if not selected_countries:
    st.info("Selecciona al menos un país en el panel izquierdo.")
else:
    df_countries = df[df["location"].isin(selected_countries)].sort_values("date")

    palette = ["#00f5d4", "#8b5cf6", "#f0abfc", "#fb923c", "#38bdf8",
               "#4ade80", "#facc15", "#f87171", "#a78bfa", "#34d399"]

    col1, col2 = st.columns(2)

    with col1:
        # Line chart — evolution
        fig_lines = go.Figure()
        for i, country in enumerate(selected_countries):
            cdf = df_countries[df_countries["location"] == country]
            fig_lines.add_trace(go.Scatter(
                x=cdf["date"], y=cdf[metric],
                name=country,
                line=dict(color=palette[i % len(palette)], width=2),
                hovertemplate=f"<b>{country}</b><br>%{{x|%d %b %Y}}<br>{metric}: %{{y:,.0f}}<extra></extra>"
            ))
        fig_lines.update_layout(
            paper_bgcolor="#04040f", plot_bgcolor="#080818",
            font=dict(family="Space Mono", color="#c8c8e8", size=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            hovermode="x unified", height=350,
            title=dict(text="Evolución acumulada", font=dict(size=12, color="#6464a0"))
        )
        st.plotly_chart(fig_lines, use_container_width=True)

    with col2:
        # Bar chart — latest values
        bar_data = latest[latest["location"].isin(selected_countries)].sort_values(metric, ascending=False)
        fig_bar = go.Figure(go.Bar(
            x=bar_data["location"],
            y=bar_data[metric],
            marker=dict(
                color=[palette[i % len(palette)] for i in range(len(bar_data))],
                opacity=0.85
            ),
            text=bar_data[metric].apply(fmt),
            textposition="outside",
            textfont=dict(size=10)
        ))
        fig_bar.update_layout(
            paper_bgcolor="#04040f", plot_bgcolor="#080818",
            font=dict(family="Space Mono", color="#c8c8e8", size=10),
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            height=350,
            title=dict(text="Total acumulado (último dato)", font=dict(size=12, color="#6464a0"))
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # New cases heatmap-style area
    st.markdown("**Nuevos casos diarios**")
    df_new = df_countries[["date", "location", "new_cases"]].dropna()
    fig_area = go.Figure()
    for i, country in enumerate(selected_countries):
        cdf = df_new[df_new["location"] == country]
        fig_area.add_trace(go.Scatter(
            x=cdf["date"], y=cdf["new_cases"].clip(lower=0),
            name=country, stackgroup=None,
            line=dict(color=palette[i % len(palette)], width=1.5),
            fill="tozeroy", fillcolor=f"rgba{tuple(list(int(palette[i%len(palette)].lstrip('#')[j:j+2],16) for j in (0,2,4))+[0.06])}"
        ))
    fig_area.update_layout(
        paper_bgcolor="#04040f", plot_bgcolor="#080818",
        font=dict(family="Space Mono", color="#c8c8e8", size=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        hovermode="x unified", height=280
    )
    st.plotly_chart(fig_area, use_container_width=True)

st.markdown("---")

# ── Section 3: Forecasting ────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">// 03 — Forecasting</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-title">Predicción con <span style="color:#00f5d4">Suavizado Exponencial</span> · {forecast_country}</div>', unsafe_allow_html=True)

fc_df = df[df["location"] == forecast_country][["date", "cases"]].dropna().sort_values("date")
fc_series = fc_df.set_index("date")["cases"]
fc_series = fc_series[fc_series > 0]

if len(fc_series) < 60:
    st.warning(f"No hay suficientes datos para {forecast_country}.")
else:
    split = int(len(fc_series) * train_pct / 100)
    train = fc_series.iloc[:split]
    test  = fc_series.iloc[split:]

    with st.spinner("Entrenando modelo Holt-Winters…"):
        model = ExponentialSmoothing(
            train, trend="add", seasonal=None,
            initialization_method="estimated"
        )
        fitted = model.fit(optimized=True)

    # Forecast on test + future
    forecast_test   = fitted.forecast(steps=len(test))
    forecast_future = fitted.forecast(steps=len(test) + forecast_steps).iloc[len(test):]

    # Metrics
    rmse = np.sqrt(mean_squared_error(test, forecast_test))
    mape = np.mean(np.abs((test.values - forecast_test.values) / test.values.clip(1))) * 100

    # KPIs
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Datos de entrenamiento", f"{len(train):,} días")
    col_m2.metric("RMSE", f"{rmse:,.0f}")
    col_m3.metric("MAPE", f"{mape:.1f}%")

    # Chart
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=train.index, y=train.values,
        name="Entrenamiento", line=dict(color="#8b5cf6", width=2),
        hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra>Train</extra>"
    ))
    fig_fc.add_trace(go.Scatter(
        x=test.index, y=test.values,
        name="Prueba real", line=dict(color="#fb923c", width=2),
        hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra>Test real</extra>"
    ))
    fig_fc.add_trace(go.Scatter(
        x=test.index, y=forecast_test.values,
        name="Predicción (test)", line=dict(color="#00f5d4", width=2, dash="dash"),
        hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra>Predicción</extra>"
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast_future.index, y=forecast_future.values,
        name=f"Forecast +{forecast_steps}d",
        line=dict(color="#f0abfc", width=2, dash="dot"),
        hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra>Forecast</extra>"
    ))
    # Confidence band (simple ±1.5*std)
    std = (test.values - forecast_test.values).std()
    upper = forecast_future.values + 1.5 * std
    lower = np.maximum(forecast_future.values - 1.5 * std, 0)
    fig_fc.add_trace(go.Scatter(
        x=list(forecast_future.index) + list(forecast_future.index[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor="rgba(240,171,252,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Intervalo de confianza", hoverinfo="skip"
    ))

    fig_fc.add_vrect(
        x0=test.index[0], x1=test.index[-1],
        fillcolor="rgba(251,146,60,0.04)",
        layer="below", line_width=0,
        annotation_text="Período de prueba",
        annotation_font=dict(color="#fb923c", size=10)
    )

    fig_fc.update_layout(
        paper_bgcolor="#04040f", plot_bgcolor="#080818",
        font=dict(family="Space Mono", color="#c8c8e8", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,245,212,0.2)", borderwidth=1),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Casos acumulados"),
        hovermode="x unified", height=420
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown(f"""
    <span class="badge">Modelo: Holt-Winters (trend=additive)</span>
    <span class="badge">RMSE: {rmse:,.0f}</span>
    <span class="badge">MAPE: {mape:.1f}%</span>
    <span class="badge">Train: {train_pct}% · Test: {100-train_pct}%</span>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Section 4: Top países ─────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">// 04 — Ranking Global</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Top 15 países por <span style="color:#00f5d4">casos confirmados</span></div>', unsafe_allow_html=True)

top15 = latest.nlargest(15, "cases")[["location", "cases", "deaths", "population"]].copy()
top15["CFR (%)"]      = (top15["deaths"] / top15["cases"] * 100).round(2)
top15["Casos / 1M"]   = (top15["cases"] / top15["population"] * 1e6).round(0).astype("Int64")
top15["cases_fmt"]    = top15["cases"].apply(fmt)
top15["deaths_fmt"]   = top15["deaths"].apply(fmt)

fig_top = go.Figure(go.Bar(
    y=top15["location"][::-1],
    x=top15["cases"][::-1],
    orientation="h",
    marker=dict(
        color=top15["CFR (%)"][::-1],
        colorscale=[[0,"#00f5d4"],[0.5,"#8b5cf6"],[1,"#f0abfc"]],
        colorbar=dict(title="CFR %", tickfont=dict(size=10))
    ),
    text=top15["cases_fmt"][::-1],
    textposition="outside",
    customdata=top15[["deaths_fmt","CFR (%)","Casos / 1M"]][::-1].values,
    hovertemplate="<b>%{y}</b><br>Casos: %{text}<br>Muertes: %{customdata[0]}<br>CFR: %{customdata[1]:.2f}%<br>Casos/1M: %{customdata[2]:,}<extra></extra>"
))
fig_top.update_layout(
    paper_bgcolor="#04040f", plot_bgcolor="#080818",
    font=dict(family="Space Mono", color="#c8c8e8", size=11),
    margin=dict(l=0, r=60, t=10, b=0),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    height=480
)
st.plotly_chart(fig_top, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(0,245,212,0.1);
     font-size:0.7rem;color:#6464a0;display:flex;justify-content:space-between;">
  <span>Fuente: Our World in Data · owid-covid-data.csv</span>
  <span>Modelo: Holt-Winters Exponential Smoothing · statsmodels</span>
</div>
""", unsafe_allow_html=True)
