import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cohort_engine import SurvivalCurve

COLORS = {
    "purple": "#7F77DD", "teal": "#1D9E75", "amber": "#EF9F27",
    "coral": "#D85A30", "gray": "#888780", "light_bg": "#F7F6F3",
    "border": "#D3D1C7", "text": "#2C2C2A", "muted": "#888780",
}

def _style():
    plt.rcParams.update({
        "figure.facecolor": COLORS["light_bg"], "axes.facecolor": COLORS["light_bg"],
        "axes.edgecolor": COLORS["border"], "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
        "text.color": COLORS["text"], "font.family": "sans-serif",
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "medium",
        "axes.titlepad": 12, "axes.labelsize": 11,
    })

def plot_ltv_distribution(result, save_path=None):
    """Plots the distribution of customer LTV across all simulations, highlighting the worst 10% tail."""
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ltv = result.ltv_array
    p10, p50, p90 = np.percentile(ltv, [10, 50, 90])
    n, bins, patches = ax.hist(ltv, bins=80, color=COLORS["purple"], alpha=0.75, edgecolor="none")
    for patch, left in zip(patches, bins[:-1]):
        if left < p10:
            patch.set_facecolor(COLORS["coral"]); patch.set_alpha(0.8)
    for pval, label, color in [
        (p10, f"P10  ${p10:,.0f}", COLORS["coral"]),
        (p50, f"P50  ${p50:,.0f}", COLORS["teal"]),
        (p90, f"P90  ${p90:,.0f}", COLORS["amber"]),
    ]:
        ax.axvline(pval, color=color, lw=1.5, linestyle="--")
        ax.text(pval + 2, ax.get_ylim()[1] * 0.85, label, color=color, fontsize=9.5, fontweight="medium")
    ax.set_xlabel("Customer LTV (discounted, $)")
    ax.set_ylabel("Frequency")
    ax.set_title("LTV distribution across 10,000 simulations\nRed tail = worst 10% of scenarios")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_revenue_fan(result, save_path=None):
    """Plots the fan chart of monthly revenue trajectories across all simulations, showing the cone of possible futures."""
    _style()
    fig, ax = plt.subplots(figsize=(11, 5))
    paths = result.monthly_revenue_paths
    months = np.arange(paths.shape[1])
    p10, p25, p50, p75, p90 = [np.percentile(paths, p, axis=0) for p in [10, 25, 50, 75, 90]]
    ax.fill_between(months, p10, p90, alpha=0.15, color=COLORS["purple"], label="P10-P90")
    ax.fill_between(months, p25, p75, alpha=0.30, color=COLORS["purple"], label="P25-P75")
    ax.plot(months, p50, color=COLORS["purple"], lw=2.0, label="P50 (median)")
    for i in np.random.choice(len(paths), 20, replace=False):
        ax.plot(months, paths[i], color=COLORS["gray"], alpha=0.08, lw=0.6)
    ax.set_xlabel("Month"); ax.set_ylabel("Monthly revenue ($)")
    ax.set_title("Revenue fan chart — cone of possible futures")
    ax.legend(frameon=False, fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_survival_curves(save_path=None):
    """Plots cohort survival curves under different churn assumptions, comparing a naive constant churn model to a more realistic modified BG model."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    months = np.arange(36)
    churn_rates = [0.03, 0.05, 0.08, 0.12]
    palette = [COLORS["teal"], COLORS["purple"], COLORS["amber"], COLORS["coral"]]
    for ax, model, title in zip(axes,
        ["constant", "modified_bg"],
        ["Constant churn (naive model)", "Modified BG survival (realistic)"]):
        for churn, color in zip(churn_rates, palette):
            s = SurvivalCurve(churn, 36, model).compute()
            ax.plot(months, s * 100, color=color, lw=2, label=f"{churn:.0%} churn")
        ax.set_xlabel("Months since acquisition"); ax.set_ylabel("% customers still active")
        ax.set_title(title); ax.legend(frameon=False, fontsize=10); ax.set_ylim(0, 105)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_sensitivity_tornado(result, save_path=None):
    """Plots a tornado chart showing which input variables (CAC, payback months, etc.) have the strongest correlation with LTV outcomes across simulations."""
    _style()
    ltv = result.ltv_array
    drivers = {
        "CAC level": result.cac_array,
        "Payback months": result.payback_array.clip(0, 36),
        "Net value (LTV-CAC)": result.net_value_array,
    }
    correlations = {}
    for name, arr in drivers.items():
        finite = np.isfinite(arr)
        if finite.sum() > 10:
            correlations[name] = np.corrcoef(arr[finite], ltv[finite])[0, 1]
    sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]))
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = [COLORS["teal"] if v > 0 else COLORS["coral"] for v in values]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.5)
    ax.axvline(0, color=COLORS["border"], lw=1)
    ax.set_xlabel("Correlation with LTV outcome")
    ax.set_title("Sensitivity — what drives LTV variance most?")
    for bar, val in zip(bars, values):
        ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_runway_survival(result, current_cash, monthly_burn, save_path=None):
    """Plots the survival curve of cash runway, showing the probability of still operating month by month under the simulated revenue paths and burn rate."""
    from risk_metrics import RunwayModel
    _style()
    rm = RunwayModel(result, current_cash, monthly_burn)
    runway = rm.compute_runway_distribution()
    months = np.arange(1, 37)
    survival_probs = [float((runway >= m).mean()) * 100 for m in months]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(months, survival_probs, color=COLORS["purple"], lw=2.5)
    ax.fill_between(months, survival_probs, alpha=0.12, color=COLORS["purple"])
    for pct, label, color in [(90, "90%", COLORS["teal"]),
                               (75, "75%", COLORS["amber"]),
                               (50, "50%", COLORS["coral"])]:
        ax.axhline(pct, color=color, lw=1.2, linestyle="--", alpha=0.8)
        crossings = [m for m, p in zip(months, survival_probs) if p <= pct]
        if crossings:
            ax.axvline(crossings[0], color=color, lw=1.0, linestyle=":", alpha=0.7)
            ax.text(crossings[0] + 0.3, pct + 2, f"Month {crossings[0]}", color=color, fontsize=9)
    ax.set_xlabel("Month"); ax.set_ylabel("P(company still operating) %")
    ax.set_title("Cash runway survival curve")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_full_dashboard(result, current_cash=5_000_000, monthly_burn=400_000, save_path=None):
    """Plots a 2x2 dashboard with: (1) LTV distribution, (2) revenue fan chart, (3) cohort survival curves, (4) cash runway survival curve."""
    from risk_metrics import RunwayModel
    _style()
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ltv = result.ltv_array
    p10, p50, p90 = np.percentile(ltv, [10, 50, 90])
    n, bins, patches = ax1.hist(ltv, bins=60, color=COLORS["purple"], alpha=0.75, edgecolor="none")
    for patch, left in zip(patches, bins[:-1]):
        if left < p10:
            patch.set_facecolor(COLORS["coral"]); patch.set_alpha(0.8)
    for pval, label, color in [(p10, f"P10 ${p10:,.0f}", COLORS["coral"]),
                                (p50, f"P50 ${p50:,.0f}", COLORS["teal"]),
                                (p90, f"P90 ${p90:,.0f}", COLORS["amber"])]:
        ax1.axvline(pval, color=color, lw=1.5, linestyle="--")
        ax1.text(pval + 1, ax1.get_ylim()[1] * 0.8, label, color=color, fontsize=8.5, fontweight="medium")
    ax1.set_title("LTV distribution"); ax1.set_xlabel("LTV ($)"); ax1.set_ylabel("Frequency")

    ax2 = fig.add_subplot(gs[0, 1])
    paths = result.monthly_revenue_paths
    months = np.arange(paths.shape[1])
    p10r, p25r, p50r, p75r, p90r = [np.percentile(paths, p, axis=0) for p in [10, 25, 50, 75, 90]]
    ax2.fill_between(months, p10r, p90r, alpha=0.15, color=COLORS["purple"])
    ax2.fill_between(months, p25r, p75r, alpha=0.30, color=COLORS["purple"])
    ax2.plot(months, p50r, color=COLORS["purple"], lw=2.0)
    ax2.set_title("Revenue fan chart"); ax2.set_xlabel("Month"); ax2.set_ylabel("Monthly revenue ($)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))

    ax3 = fig.add_subplot(gs[1, 0])
    for churn, color, label in [(0.03, COLORS["teal"], "3%"), (0.05, COLORS["purple"], "5%"),
                                  (0.08, COLORS["amber"], "8%"), (0.12, COLORS["coral"], "12%")]:
        s = SurvivalCurve(churn, 36, "modified_bg").compute()
        ax3.plot(np.arange(36), s * 100, color=color, lw=2, label=f"{label} churn")
    ax3.set_title("Cohort survival curves"); ax3.set_xlabel("Months"); ax3.set_ylabel("% active")
    ax3.legend(frameon=False, fontsize=9)

    ax4 = fig.add_subplot(gs[1, 1])
    rm = RunwayModel(result, current_cash, monthly_burn)
    runway = rm.compute_runway_distribution()
    months_r = np.arange(1, 37)
    sp = [float((runway >= m).mean()) * 100 for m in months_r]
    ax4.plot(months_r, sp, color=COLORS["purple"], lw=2.5)
    ax4.fill_between(months_r, sp, alpha=0.12, color=COLORS["purple"])
    for pct, color in [(90, COLORS["teal"]), (75, COLORS["amber"]), (50, COLORS["coral"])]:
        ax4.axhline(pct, color=color, lw=1.2, linestyle="--", alpha=0.8)
    ax4.set_title("Cash runway survival"); ax4.set_xlabel("Month"); ax4.set_ylabel("P(operating) %")
    ax4.set_ylim(0, 105)

    plt.suptitle("MonteMetrics — Board Dashboard", fontsize=15, fontweight="medium", y=1.01)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Dashboard saved: {save_path}")
    plt.close()