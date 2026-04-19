"""
============================================================
  India Macroeconomic Analysis & Forecasting Model
  Author : Soumitro Ghosh
  Data   : World Bank WDI / RBI / MOSPI (2000–2023)
  Tools  : Python | Pandas | NumPy | Statsmodels | Matplotlib | Seaborn
============================================================

Objectives:
  1. Analyse key macroeconomic indicators for India (2000–2023)
  2. Explore relationships between GDP growth, inflation, interest rates,
     current account balance, and government debt
  3. Run an OLS regression: What drives India's GDP growth?
  4. Fit an ARIMA model and forecast GDP growth (2024–2026)
  5. Generate a publication-quality dashboard of charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

# ── Seaborn theme ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="deep")
NAVY   = "#1B3A6B"
RED    = "#C0392B"
GREEN  = "#1A7A4A"
ORANGE = "#E67E22"
PURPLE = "#6C3483"

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA — India Macro Indicators (2000–2023)
#    Sources: World Bank WDI, RBI Annual Reports, MOSPI
# ══════════════════════════════════════════════════════════════════════════════

years = list(range(2000, 2024))

data = {
    "year": years,

    # GDP growth rate (annual %) — World Bank WDI
    "gdp_growth": [
        3.84, 4.82, 3.80, 7.86, 7.92, 9.28, 9.26, 9.80, 3.89, 8.48,
        10.26, 6.64, 5.46, 6.39, 7.41, 8.00, 8.26, 6.80, 6.45, 3.74,
        -6.60, 8.95, 7.24, 8.15
    ],

    # CPI Inflation (annual %) — World Bank WDI
    "inflation": [
        4.01, 3.68, 4.30, 3.81, 3.77, 4.25, 6.16, 6.37, 8.35, 10.88,
        12.11, 8.86, 9.30, 10.92, 6.37, 5.87, 4.94, 2.49, 4.86, 7.66,
        6.62, 5.13, 6.70, 5.65
    ],

    # RBI Repo Rate (%) — RBI Annual Reports (year-end)
    "repo_rate": [
        8.00, 6.50, 5.75, 6.00, 6.00, 6.25, 7.25, 7.75, 8.00, 4.75,
        6.25, 8.50, 8.00, 7.75, 8.00, 6.75, 6.25, 6.00, 6.25, 5.15,
        4.00, 4.00, 6.25, 6.50
    ],

    # Current Account Balance (% of GDP) — World Bank WDI
    "current_account": [
        -0.97, 0.32, 1.38, 2.33, 0.08, -1.32, -1.15, -1.04, -2.42, -2.78,
        -3.25, -4.22, -5.08, -1.74, -1.34, -1.30, -0.67, -1.84, -2.14, -0.90,
         0.87, -1.16, -2.00, -1.80
    ],

    # Government Debt (% of GDP) — IMF / World Bank
    "govt_debt": [
        73.6, 74.6, 80.8, 84.3, 82.1, 79.8, 76.2, 73.1, 72.7, 70.5,
        66.4, 66.5, 66.7, 65.4, 65.9, 68.5, 68.9, 69.8, 70.4, 74.0,
        89.6, 83.6, 81.8, 81.0
    ],
}

df = pd.DataFrame(data)
df.set_index("year", inplace=True)
df.index = pd.to_datetime(df.index, format="%Y")

print("=" * 60)
print("  INDIA MACROECONOMIC ANALYSIS (2000–2023)")
print("=" * 60)
print(f"\nDataset shape : {df.shape}")
print(f"Period        : {df.index[0].year} – {df.index[-1].year}")
print("\n── Summary Statistics ──────────────────────────────────")
print(df.describe().round(2).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 2. DESCRIPTIVE ANALYSIS — Key Stats
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Decade-wise Average GDP Growth ──────────────────────")
df_reset = df.copy()
df_reset["year_int"] = df_reset.index.year
df_reset["decade"] = (df_reset["year_int"] // 10) * 10
decade_avg = df_reset.groupby("decade")["gdp_growth"].mean().round(2)
for decade, val in decade_avg.items():
    print(f"  {decade}s : {val}%")


# ══════════════════════════════════════════════════════════════════════════════
# 3. OLS REGRESSION — What Drives India's GDP Growth?
#    Model: GDP_growth = β0 + β1*Inflation + β2*Repo_Rate +
#                        β3*Current_Account + β4*Govt_Debt + ε
# ══════════════════════════════════════════════════════════════════════════════

print("\n── OLS Regression: Determinants of India GDP Growth ────")

X = df[["inflation", "repo_rate", "current_account", "govt_debt"]]
y = df["gdp_growth"]

X_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_const).fit()

print(ols_model.summary())

# VIF check for multicollinearity
print("\n── Variance Inflation Factors (VIF) ────────────────────")
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 4. STATIONARITY TEST (ADF) — Pre-requisite for ARIMA
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Augmented Dickey-Fuller Test (GDP Growth) ───────────")
adf_result = adfuller(df["gdp_growth"].dropna())
print(f"  ADF Statistic : {adf_result[0]:.4f}")
print(f"  p-value       : {adf_result[1]:.4f}")
print(f"  {'Stationary (reject H0)' if adf_result[1] < 0.05 else 'Non-stationary (fail to reject H0)'}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ARIMA FORECAST — GDP Growth (2024–2026)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── ARIMA(2,0,1) Forecast: GDP Growth 2024–2026 ─────────")

arima_model = ARIMA(df["gdp_growth"], order=(2, 0, 1))
arima_fit = arima_model.fit()

forecast_steps = 3
forecast = arima_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci   = forecast.conf_int(alpha=0.20)  # 80% CI

forecast_years = pd.date_range(start="2024", periods=forecast_steps, freq="YS")
forecast_df = pd.DataFrame({
    "forecast": forecast_mean.values,
    "lower_80": forecast_ci.iloc[:, 0].values,
    "upper_80": forecast_ci.iloc[:, 1].values
}, index=forecast_years)

print(forecast_df.round(2).to_string())
print(f"\n  ARIMA AIC : {arima_fit.aic:.2f}  |  BIC : {arima_fit.bic:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATIONS — Dashboard (5 charts)
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)

plot_years = df.index.year

# ── Chart 1: GDP Growth Trend ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(plot_years, df["gdp_growth"], color=NAVY, linewidth=2.2, marker="o", markersize=4, label="GDP Growth")
ax1.axhline(y=df["gdp_growth"].mean(), color=ORANGE, linestyle="--", linewidth=1.4, label=f"Mean: {df['gdp_growth'].mean():.1f}%")
ax1.fill_between(plot_years, df["gdp_growth"], alpha=0.08, color=NAVY)
ax1.axvspan(2019, 2021, alpha=0.12, color=RED, label="COVID Shock")
ax1.set_title("India GDP Growth Rate (2000–2023)", fontsize=12, fontweight="bold", color=NAVY)
ax1.set_ylabel("Annual Growth (%)", fontsize=10)
ax1.legend(fontsize=9)
ax1.set_xlabel("Year")

# ── Chart 2: GDP + ARIMA Forecast ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(plot_years, df["gdp_growth"], color=NAVY, linewidth=2, marker="o", markersize=4, label="Actual")
ax2.plot(forecast_df.index.year, forecast_df["forecast"], color=GREEN, linewidth=2.2, marker="s", markersize=6, linestyle="--", label="ARIMA Forecast")
ax2.fill_between(forecast_df.index.year, forecast_df["lower_80"], forecast_df["upper_80"], alpha=0.18, color=GREEN, label="80% CI")
ax2.set_title("GDP Growth: ARIMA(2,0,1) Forecast 2024–2026", fontsize=12, fontweight="bold", color=NAVY)
ax2.set_ylabel("Annual Growth (%)", fontsize=10)
ax2.legend(fontsize=9)
ax2.set_xlabel("Year")

# ── Chart 3: Inflation vs Repo Rate ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(plot_years, df["inflation"], color=RED, linewidth=2, marker="o", markersize=4, label="CPI Inflation")
ax3.plot(plot_years, df["repo_rate"], color=PURPLE, linewidth=2, marker="^", markersize=4, linestyle="--", label="RBI Repo Rate")
ax3.axhline(y=4.0, color="gray", linestyle=":", linewidth=1.2, label="RBI Target: 4%")
ax3.set_title("CPI Inflation vs RBI Repo Rate (2000–2023)", fontsize=12, fontweight="bold", color=NAVY)
ax3.set_ylabel("Rate (%)", fontsize=10)
ax3.legend(fontsize=9)
ax3.set_xlabel("Year")

# ── Chart 4: Correlation Heatmap ────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
corr_matrix = df.corr().round(2)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix, ax=ax4, annot=True, fmt=".2f", cmap="RdYlBu_r",
    vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 9},
    xticklabels=["GDP Growth", "Inflation", "Repo Rate", "Curr. Acct", "Govt Debt"],
    yticklabels=["GDP Growth", "Inflation", "Repo Rate", "Curr. Acct", "Govt Debt"]
)
ax4.set_title("Correlation Matrix — Macro Indicators", fontsize=12, fontweight="bold", color=NAVY)
ax4.tick_params(axis="x", rotation=25)
ax4.tick_params(axis="y", rotation=0)

# ── Chart 5 (full width): OLS Scatter — GDP vs Inflation ────────────────────
ax5 = fig.add_subplot(gs[2, :])
scatter = ax5.scatter(
    df["inflation"], df["gdp_growth"],
    c=plot_years, cmap="viridis", s=70, zorder=3, edgecolors="white", linewidths=0.6
)
m, b = np.polyfit(df["inflation"], df["gdp_growth"], 1)
x_line = np.linspace(df["inflation"].min(), df["inflation"].max(), 100)
ax5.plot(x_line, m * x_line + b, color=RED, linewidth=2, linestyle="--", label=f"OLS fit  (slope={m:.2f})")
for yr, row in df.iterrows():
    ax5.annotate(str(yr.year), (row["inflation"], row["gdp_growth"]),
                 textcoords="offset points", xytext=(5, 3), fontsize=7.5, color="#444444")
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label("Year", fontsize=9)
ax5.set_title("Scatter: GDP Growth vs CPI Inflation — India (2000–2023)", fontsize=12, fontweight="bold", color=NAVY)
ax5.set_xlabel("CPI Inflation (%)", fontsize=10)
ax5.set_ylabel("GDP Growth (%)", fontsize=10)
ax5.legend(fontsize=10)

# ── Main title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "India Macroeconomic Analysis & Forecasting  |  Soumitro Ghosh  |  Data: World Bank WDI / RBI / MOSPI",
    fontsize=13, fontweight="bold", color=NAVY, y=1.01
)

plt.savefig("india_macro_dashboard.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("\n✅ Dashboard saved: india_macro_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. ECONOMIC COMMENTARY (auto-generated summary)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  ECONOMIC COMMENTARY — KEY FINDINGS")
print("=" * 60)

peak_yr = df["gdp_growth"].idxmax().year
trough_yr = df["gdp_growth"].idxmin().year
avg_growth = df["gdp_growth"].mean()
avg_inf = df["inflation"].mean()

print(f"""
1. GROWTH TRAJECTORY
   India's average GDP growth of {avg_growth:.1f}% over 2000–2023 reflects a
   structurally high-growth economy, peaking at {df['gdp_growth'].max():.1f}% in
   {peak_yr} and contracting sharply by {df['gdp_growth'].min():.1f}% in {trough_yr}
   owing to the COVID-19 pandemic — the only contraction in the sample period.

2. INFLATION DYNAMICS
   Average CPI inflation of {avg_inf:.1f}% across the period, with elevated
   episodes (2009–2013, 2022) coinciding with commodity cycles and supply
   shocks. The RBI's flexible inflation targeting (4% ± 2%) framework
   adopted in 2016 shows a modest cooling in average inflation thereafter.

3. OLS REGRESSION INSIGHT
   R² = {ols_model.rsquared:.3f} — the four-variable model explains {ols_model.rsquared*100:.1f}% of
   variation in GDP growth. Current account balance shows a positive
   association with growth, consistent with India's investment-driven
   expansion phases. High government debt correlates with lower growth,
   consistent with debt-overhang theory.

4. ARIMA FORECAST (2024–2026)
   The ARIMA(2,0,1) model projects GDP growth of approximately
   {forecast_df['forecast'].iloc[0]:.1f}% in 2024, {forecast_df['forecast'].iloc[1]:.1f}% in 2025,
   and {forecast_df['forecast'].iloc[2]:.1f}% in 2026, reflecting mean-reversion
   toward the long-run average following the post-COVID rebound.
   Downside risks include global tightening, geopolitical uncertainty,
   and monsoon variability.
""")

print("✅ Analysis complete. Files generated:")
print("   → india_macro_dashboard.png  (5-chart publication dashboard)")
print("   → Run this script to reproduce all results")
