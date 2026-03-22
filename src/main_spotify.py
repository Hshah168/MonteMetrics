import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from spotify_2025_data import fetch_spotify_live, get_spotify_subscription_metrics
from distributions import (ChurnDistribution, ARPUDistribution,
                            CACDistribution, MacroShockDistribution)
from cohort_engine import CohortLTVEngine, CohortConfig
from simulator import SimulationConfig, MonteCarloEngine
from risk_metrics import RevenueAtRisk, RunwayModel, CACBudgetOptimizer
from visualizer import (plot_ltv_distribution, plot_revenue_fan,
                         plot_survival_curves, plot_sensitivity_tornado,
                         plot_runway_survival, plot_full_dashboard)


def sep(title):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")


sep("FETCHING SPOTIFY DATA")
spot_live = fetch_spotify_live()
spot_subs = get_spotify_subscription_metrics()
live      = spot_live["live"]

ARPU        = spot_subs["monthly_arpu_usd"]          
CHURN_MEAN  = spot_subs["monthly_churn_estimate"]     
GM          = spot_subs["gross_margin"]              
CAC         = spot_subs["cac_estimated_usd"]         
NEW_SUBS_MO = spot_subs["avg_monthly_net_adds_2025"]  
CASH        = spot_subs["cash_end_2025_eur"] * 1.1575   
MONTHLY_BURN = 377_000_000 


sep("FITTING DISTRIBUTIONS TO SPOTIFY DATA")


churn_dist = ChurnDistribution(
    alpha=2.0,
    beta=164.0,
    recession_multiplier=spot_subs["recession_multiplier"]
)


arpu_dist = ARPUDistribution(
    mu=1.578,
    sigma=0.45,
    gross_margin=GM
)


cac_dist = CACDistribution(
    shape=6.0,
    scale=round(CAC / 6.0, 2)
)

macro_dist = MacroShockDistribution(
    recession_prob=spot_subs["recession_prob_annual"]
)

print(f"\n  Company        : Spotify Technology S.A. (NYSE: SPOT)")
print(f"  Data period    : Full Year 2025")
print(f"\n  Churn mean     : {churn_dist.mean:.2%}  (target: {CHURN_MEAN:.2%})")
print(f"  ARPU mean      : ${arpu_dist.expected_value:.2f}  (target: ${ARPU:.2f})")
print(f"  CAC mean       : ${cac_dist.mean:.2f}  (target: ${CAC:.2f})")
print(f"  Gross margin   : {GM:.1%}  (disclosed — record high)")

# Cohort LTV analysis
sep("SPOTIFY — Cohort LTV Analysis")

engine = CohortLTVEngine(n_months=36, discount_rate_annual=0.10)

print(f"\n── Spotify LTV at different churn scenarios ──")
print(f"   ARPU=${ARPU:.2f}, GM={GM:.1%}, CAC=${CAC:.2f}\n")

rows = []
for churn in [0.008, 0.012, 0.015, 0.020, 0.030]:
    r  = engine.compute_cohort_ltv(churn, ARPU, GM, "modified_bg")
    pb = engine.compute_cac_payback(CAC, churn, ARPU, GM)
    rows.append({
        "monthly_churn":  f"{churn:.1%}",
        "annual_churn":   f"{1-(1-churn)**12:.0%}",
        "ltv_$":          round(r["ltv"], 2),
        "ltv_cac_ratio":  round(r["ltv"] / CAC, 1),
        "payback_months": round(pb, 1) if np.isfinite(pb) else "never",
    })

print(pd.DataFrame(rows).to_string(index=False))
print(f"\n  Base case: 1.2% churn → LTV/CAC = "
      f"{engine.compute_cohort_ltv(0.012, ARPU, GM)['ltv']/CAC:.1f}x")

cohorts = [
    CohortConfig("Q1-2025", 0,  562_500, 0.012, ARPU, GM, CAC),
    CohortConfig("Q2-2025", 3,  600_000, 0.012, ARPU, GM, CAC),
    CohortConfig("Q3-2025", 6,  637_500, 0.011, ARPU, GM, CAC),
    CohortConfig("Q4-2025", 9,  750_000, 0.010, ARPU, GM, CAC),
]

schedule = engine.build_revenue_schedule(cohorts)
print("\n── Revenue schedule: first 8 months (scaled 1:4000) ──")
print(schedule[["month", "total_revenue"]].head(8).to_string(index=False))
schedule.to_csv(f"{OUTPUT_DIR}/spotify_revenue_schedule.csv", index=False)

plot_survival_curves(save_path=f"{OUTPUT_DIR}/spotify_survival_curves.png")
print("Saved: spotify_2025_survival_curves.png")


# Monte Carlo simulation
sep("MONTE CARLO SIMULATION — Spotify , 10,000 runs")

cfg = SimulationConfig(
    n_simulations=10_000,
    n_months=36,
    new_customers_per_month=NEW_SUBS_MO,
    discount_rate_annual=0.10,
    churn_dist=churn_dist,
    arpu_dist=arpu_dist,
    cac_dist=cac_dist,
    macro_dist=macro_dist,
    seed=42,
)

mc     = MonteCarloEngine(cfg)
result = mc.run(verbose=True)

df_sims = pd.DataFrame({
    "ltv":            result.ltv_array,
    "cac":            result.cac_array,
    "payback_months": result.payback_array.clip(0, 999),
    "net_value":      result.net_value_array,
    "recession_hit":  result.recession_hit_array,
})
df_sims.to_csv(f"{OUTPUT_DIR}/spotify_simulation_results.csv", index=False)

print("\n── Revenue at Risk ──")
rar = RevenueAtRisk(result)
print(rar.full_report().to_string())

print("\n── Recession stress scenario ──")
for k, v in rar.stress_scenario().items():
    print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

print("\n── Churn sensitivity ──")
print(mc.run_sensitivity("churn_mean",
      [0.008, 0.012, 0.015, 0.020, 0.030]).round(2).to_string(index=False))

print("\n── ARPU sensitivity ──")
print(mc.run_sensitivity("arpu_mean",
      [3.50, 4.00, 5.36, 6.00, 8.00]).round(2).to_string(index=False))
print("  Note: $8.00 = potential premium tier | $3.50 = free-to-paid conversion")

plot_ltv_distribution(result,    f"{OUTPUT_DIR}/spotify_ltv_distribution.png")
plot_revenue_fan(result,          f"{OUTPUT_DIR}/spotify_revenue_fan.png")
plot_sensitivity_tornado(result,  f"{OUTPUT_DIR}/spotify_sensitivity_tornado.png")
print("\nSaved: ltv_distribution, revenue_fan, sensitivity_tornado")


# Capital allocation
sep("SPOTIFY — Capital Allocation")

rm = RunwayModel(result, CASH, MONTHLY_BURN, new_customers_per_month=NEW_SUBS_MO)
print("\n Runway analysis")
for k, v in rm.fundraise_trigger_analysis().items():
    if "prob" in k:
        print(f"  {k}: {v:.1%}")
    elif "months" in k:
        print(f"  {k}: {v:.1f} months")
    else:
        print(f"  {k}: {v}")

opt = CACBudgetOptimizer(result, CASH, MONTHLY_BURN)
print("\n Max sustainable CAC ")
for k, v in opt.max_cac(new_customers_per_month=NEW_SUBS_MO).items():
    print(f"  {k}: {v}")

plot_runway_survival(result, CASH, MONTHLY_BURN,
                     f"{OUTPUT_DIR}/spotify_runway_survival.png")
plot_full_dashboard(result, CASH, MONTHLY_BURN,
                    f"{OUTPUT_DIR}/spotify_board_dashboard.png")
print("\nSaved: runway_survival, board_dashboard")


# Market comparison
sep("LIVE MARKET DATA vs MODEL OUTPUT")

p10 = result.percentile("ltv_array", 10)
p50 = result.percentile("ltv_array", 50)
p90 = result.percentile("ltv_array", 90)

print(f"\n  === Spotify live market data (yfinance) ===")
print(f"  Stock price     : ${live['current_price']:,.2f}")
print(f"  Market cap      : ${live['market_cap']/1e9:.1f}B")
print(f"  Revenue (TTM)   : ${live['revenue_ttm']/1e9:.1f}B")

print(f"\n  === Monte Carlo output ===")
print(f"  P10 LTV         : ${p10:.2f}")
print(f"  P50 LTV         : ${p50:.2f}  ← base case")
print(f"  P90 LTV         : ${p90:.2f}")
print(f"  LTV/CAC (P50)   : {p50/CAC:.1f}x")

total_subs = spot_subs["premium_subscribers_end_2025"]
implied    = p50 * total_subs
premium    = live["market_cap"] - implied

print(f"\n  === Implied value check ===")
print(f"  P50 LTV x 290M subs = ${implied/1e9:.1f}B implied subscriber value")
print(f"  vs Spotify market cap = ${live['market_cap']/1e9:.1f}B")
if implied > 0:
    print(f"  Market pays {live['market_cap']/implied:.1f}x pure subscription LTV")
print(f"  Premium over sub value = ${premium/1e9:.1f}B")
print(f"  (Brand + podcast + ads + AI audio + live events)")


sep("COMPLETE — All Spotify outputs saved")
print(f"\n  Company   : Spotify Technology S.A. (NYSE: SPOT)")
print(f"  Data      : FY2025 earnings")
print(f"  Subs      : 290M premium | 751M MAU")
print(f"  ARPU      : ${ARPU:.2f}/month")
print(f"  Churn     : {CHURN_MEAN:.1%}/month (estimated)\n")

for f in sorted(os.listdir(OUTPUT_DIR)):
    if "spotify" in f:
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
        print(f"  {f}  ({size} KB)")