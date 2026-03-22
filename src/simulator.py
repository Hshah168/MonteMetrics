import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field
from distributions import (ChurnDistribution, ARPUDistribution,
                           CACDistribution, MacroShockDistribution)
from cohort_engine import SurvivalCurve


@dataclass
class SimulationConfig:
    n_simulations: int = 10_000
    n_months: int = 36
    new_customers_per_month: int = 500
    discount_rate_annual: float = 0.12
    churn_dist: ChurnDistribution = field(default_factory=ChurnDistribution)
    arpu_dist: ARPUDistribution = field(default_factory=ARPUDistribution)
    cac_dist: CACDistribution = field(default_factory=CACDistribution)
    macro_dist: MacroShockDistribution = field(default_factory=MacroShockDistribution)
    seed: Optional[int] = 42


@dataclass
class SimulationResult:
    ltv_array: np.ndarray
    cac_array: np.ndarray
    payback_array: np.ndarray
    total_revenue_array: np.ndarray
    net_value_array: np.ndarray
    recession_hit_array: np.ndarray
    monthly_revenue_paths: np.ndarray

    def percentile(self, metric: str, p: float) -> float:
        return float(np.percentile(getattr(self, metric), p))

    def summary(self) -> pd.DataFrame:
        metrics = {
            "ltv": self.ltv_array,
            "cac": self.cac_array,
            "payback_months": self.payback_array,
            "net_value": self.net_value_array,
            "total_revenue": self.total_revenue_array,
        }
        rows = []
        for name, arr in metrics.items():
            finite = arr[np.isfinite(arr)]
            rows.append({
                "metric": name,
                "mean": np.mean(finite),
                "std": np.std(finite),
                "p5": np.percentile(finite, 5),
                "p10": np.percentile(finite, 10),
                "p25": np.percentile(finite, 25),
                "p50": np.percentile(finite, 50),
                "p75": np.percentile(finite, 75),
                "p90": np.percentile(finite, 90),
                "p95": np.percentile(finite, 95),
            })
        return pd.DataFrame(rows).set_index("metric").round(2)


class MonteCarloEngine:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def run(self, verbose: bool = True) -> SimulationResult:
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        cfg = self.config
        N = cfg.n_simulations
        M = cfg.n_months

        if verbose:
            print(f"Running {N:,} simulations x {M} months...")

        churn_samples = cfg.churn_dist.sample(N)
        arpu_samples = cfg.arpu_dist.sample(N)
        cac_samples = cfg.cac_dist.sample(N)
        macro_samples = cfg.macro_dist.sample_monthly(N, M)

        recession_in_first_year = macro_samples[:, :12].max(axis=1)
        churn_effective = churn_samples * (
            1 + (cfg.churn_dist.recession_multiplier - 1) * recession_in_first_year
        )
        churn_effective = np.clip(churn_effective, 0, 1)

        monthly_discount = (1 + cfg.discount_rate_annual) ** (1 / 12) - 1
        discount_factors = np.array(
            [(1 + monthly_discount) ** (-t) for t in range(M)]
        )

        t_arr = np.arange(M)
        slowdown = np.exp(-0.03 * t_arr)
        survival = np.zeros((N, M))
        survival[:, 0] = 1.0
        for t in range(1, M):
            effective_churn_t = churn_effective * slowdown[t]
            survival[:, t] = survival[:, t - 1] * (1 - effective_churn_t)

        monthly_gp = arpu_samples * cfg.arpu_dist.gross_margin
        revenue_per_customer = survival * monthly_gp[:, np.newaxis]
        total_monthly_revenue = revenue_per_customer * cfg.new_customers_per_month
        discounted_revenue = revenue_per_customer * discount_factors[np.newaxis, :]
        ltv = discounted_revenue.sum(axis=1)
        total_revenue = total_monthly_revenue.sum(axis=1)
        net_value = ltv - cac_samples

        cumulative_gp = revenue_per_customer.cumsum(axis=1)
        payback = np.full(N, np.inf)
        for t in range(M):
            mask = (cumulative_gp[:, t] >= cac_samples) & np.isinf(payback)
            payback[mask] = float(t + 1)

        if verbose:
            result = SimulationResult(
                ltv_array=ltv, cac_array=cac_samples,
                payback_array=payback, total_revenue_array=total_revenue,
                net_value_array=net_value, recession_hit_array=recession_in_first_year,
                monthly_revenue_paths=total_monthly_revenue,
            )
            print("Done.\n")
            print(result.summary().to_string())

        return SimulationResult(
            ltv_array=ltv, cac_array=cac_samples,
            payback_array=payback, total_revenue_array=total_revenue,
            net_value_array=net_value, recession_hit_array=recession_in_first_year,
            monthly_revenue_paths=total_monthly_revenue,
        )

    def run_sensitivity(self, param: str, values: list) -> pd.DataFrame:
        rows = []
        orig_seed = self.config.seed
        for val in values:
            cfg = self.config
            if param == "churn_mean":
                total = cfg.churn_dist.alpha + cfg.churn_dist.beta
                cfg.churn_dist.alpha = val * total
                cfg.churn_dist.beta = (1 - val) * total
            elif param == "arpu_mean":
                cfg.arpu_dist.mu = np.log(val) - 0.5 * cfg.arpu_dist.sigma ** 2
            elif param == "recession_prob":
                cfg.macro_dist.recession_prob = val
            self.config.seed = orig_seed
            result = self.run(verbose=False)
            finite_payback = result.payback_array[np.isfinite(result.payback_array)]
            rows.append({
                param: val,
                "p10_ltv": result.percentile("ltv_array", 10),
                "p50_ltv": result.percentile("ltv_array", 50),
                "p90_ltv": result.percentile("ltv_array", 90),
                "p50_payback": float(np.median(finite_payback)) if len(finite_payback) else np.nan,
            })
        self.config.seed = orig_seed
        return pd.DataFrame(rows)