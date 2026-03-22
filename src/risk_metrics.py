import numpy as np
import pandas as pd
from typing import Optional
import copy


class RevenueAtRisk:
    def __init__(self, result, baseline_revenue: Optional[float] = None):
        self.result = result
        self.baseline = baseline_revenue or float(np.median(result.total_revenue_array))

    def var(self, confidence: float = 0.95) -> float:
        cutoff = np.percentile(self.result.total_revenue_array, (1 - confidence) * 100)
        return self.baseline - cutoff

    def cvar(self, confidence: float = 0.95) -> float:
        cutoff = np.percentile(self.result.total_revenue_array, (1 - confidence) * 100)
        tail = self.result.total_revenue_array[self.result.total_revenue_array <= cutoff]
        if len(tail) == 0:
            return 0.0
        return self.baseline - float(tail.mean())

    def stress_scenario(self) -> dict:
        mask = self.result.recession_hit_array == 1.0
        recession_rev = self.result.total_revenue_array[mask]
        normal_rev = self.result.total_revenue_array[~mask]
        return {
            "recession_p50_revenue": float(np.median(recession_rev)) if len(recession_rev) else np.nan,
            "normal_p50_revenue": float(np.median(normal_rev)) if len(normal_rev) else np.nan,
            "recession_impact_pct": float(
                (np.median(normal_rev) - np.median(recession_rev)) / np.median(normal_rev) * 100
            ) if len(recession_rev) and len(normal_rev) else np.nan,
            "pct_sims_in_recession": float(mask.mean() * 100),
        }

    def full_report(self) -> pd.DataFrame:
        rows = []
        for conf in [0.90, 0.95, 0.99]:
            rows.append({
                "confidence": f"{conf:.0%}",
                "var_revenue": round(self.var(conf), 2),
                "cvar_revenue": round(self.cvar(conf), 2),
                "threshold_revenue": round(self.baseline - self.var(conf), 2),
            })
        return pd.DataFrame(rows)


class RunwayModel:
    def __init__(self, result, current_cash: float,
                 monthly_burn_non_revenue: float,
                 new_customers_per_month: int = 500):
        self.result = result
        self.current_cash = current_cash
        self.monthly_burn = monthly_burn_non_revenue
        self.new_customers = new_customers_per_month

    def compute_runway_distribution(self) -> np.ndarray:
        N, M = self.result.monthly_revenue_paths.shape
        runway = np.full(N, float(M))
        cash = np.full(N, float(self.current_cash))
        cac_total_monthly = self.result.cac_array * self.new_customers
        for t in range(M):
            revenue = self.result.monthly_revenue_paths[:, t]
            net_cashflow = revenue - self.monthly_burn - cac_total_monthly
            cash += net_cashflow
            hit_zero = (cash <= 0) & (runway == M)
            runway[hit_zero] = float(t + 1)
        return runway

    def survival_probability(self, target_months: int) -> float:
        runway = self.compute_runway_distribution()
        return float((runway >= target_months).mean())

    def fundraise_trigger_analysis(self, target_runway_floor: int = 12,
                                   target_confidence: float = 0.90) -> dict:
        runway = self.compute_runway_distribution()
        p10_runway = float(np.percentile(runway, 10))
        p50_runway = float(np.median(runway))
        fundraise_lead_time = 6
        recommended_start = max(1, p10_runway - target_runway_floor - fundraise_lead_time)
        return {
            "p10_runway_months": p10_runway,
            "p50_runway_months": p50_runway,
            "p90_runway_months": float(np.percentile(runway, 90)),
            "recommended_fundraise_start_month": recommended_start,
            "survival_prob_12mo": self.survival_probability(12),
            "survival_prob_18mo": self.survival_probability(18),
            "survival_prob_24mo": self.survival_probability(24),
        }


class CACBudgetOptimizer:
    def __init__(self, result, current_cash: float, monthly_burn: float):
        self.result = result
        self.current_cash = current_cash
        self.monthly_burn = monthly_burn

    def max_cac(self, min_runway_months: int = 12,
                confidence: float = 0.90,
                new_customers_per_month: int = 500) -> dict:
        low, high = 50.0, 800.0
        best_cac = low
        for _ in range(20):
            mid = (low + high) / 2
            test_result = copy.copy(self.result)
            test_result.cac_array = np.full_like(self.result.cac_array, mid)
            rm = RunwayModel(test_result, self.current_cash,
                             self.monthly_burn, new_customers_per_month)
            if rm.survival_probability(min_runway_months) >= confidence:
                best_cac = mid
                low = mid
            else:
                high = mid
            if high - low < 1.0:
                break
        return {
            "max_sustainable_cac": round(best_cac, 2),
            "constraint": f"P(runway>{min_runway_months}mo) >= {confidence:.0%}",
            "current_mean_cac": float(self.result.cac_array.mean()),
            "headroom_pct": round((best_cac / self.result.cac_array.mean() - 1) * 100, 1),
        }