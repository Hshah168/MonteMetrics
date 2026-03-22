import numpy as np
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class CohortConfig:
    name: str
    acquisition_month: int
    n_customers: int
    monthly_churn: float
    arpu: float
    gross_margin: float = 0.72
    acquisition_cost_per_customer: float = 180.0


class SurvivalCurve:
    """Computes the survival curve for a cohort based on monthly churn and a specified model."""
    def __init__(self, monthly_churn: float, n_months: int, model: str = "modified_bg"):
        self.monthly_churn = monthly_churn
        self.n_months = n_months
        self.model = model

    def compute(self) -> np.ndarray:
        if self.model == "constant":
            return self._constant_churn()
        elif self.model == "modified_bg":
            return self._modified_bg()
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def _constant_churn(self) -> np.ndarray:
        t = np.arange(self.n_months)
        return (1 - self.monthly_churn) ** t

    def _modified_bg(self) -> np.ndarray:
        c = self.monthly_churn
        a = 0.65
        b = max(a * (1 - c) / c, 0.01)
        survival = np.ones(self.n_months)
        for t in range(1, self.n_months):
            survival[t] = survival[t - 1] * (b + t - 1) / (a + b + t - 1)
        return survival


class CohortLTVEngine:
    def __init__(self, n_months: int = 36, discount_rate_annual: float = 0.12):
        self.n_months = n_months
        self.monthly_discount = (1 + discount_rate_annual) ** (1 / 12) - 1

    def compute_cohort_ltv(self, monthly_churn: float, arpu: float,
                           gross_margin: float = 0.72,
                           survival_model: str = "modified_bg") -> dict:
        """Computes the LTV for a cohort given its monthly churn, ARPU, and gross margin, using the specified survival model."""
        sc = SurvivalCurve(monthly_churn, self.n_months, model=survival_model)
        survival = sc.compute()
        monthly_gp = arpu * gross_margin
        discount_factors = np.array(
            [(1 + self.monthly_discount) ** (-t) for t in range(self.n_months)]
        )
        monthly_revenue = survival * monthly_gp
        discounted_revenue = monthly_revenue * discount_factors
        ltv = discounted_revenue.sum()
        return {
            "ltv": ltv,
            "undiscounted_ltv": monthly_revenue.sum(),
            "survival": survival,
            "monthly_revenue": monthly_revenue,
            "discounted_revenue": discounted_revenue,
            "months_active_expected": survival.sum(),
        }

    def build_revenue_schedule(self, cohorts: List[CohortConfig],
                               monthly_churn_override: Optional[float] = None,
                               arpu_override: Optional[float] = None) -> pd.DataFrame:
        """Builds a month-by-month revenue schedule for multiple cohorts, allowing for optional overrides of churn and ARPU."""
        max_month = self.n_months
        revenue_matrix = np.zeros((max_month, len(cohorts)))
        for j, cohort in enumerate(cohorts):
            churn = monthly_churn_override or cohort.monthly_churn
            arpu = arpu_override or cohort.arpu
            sc = SurvivalCurve(churn, max_month, model="modified_bg")
            survival = sc.compute()
            monthly_gp = arpu * cohort.gross_margin
            for t in range(max_month):
                months_since = t - cohort.acquisition_month
                if months_since < 0 or months_since >= len(survival):
                    continue
                revenue_matrix[t, j] = cohort.n_customers * survival[months_since] * monthly_gp
        df = pd.DataFrame(revenue_matrix, columns=[c.name for c in cohorts])
        df["total_revenue"] = df.sum(axis=1)
        df["month"] = range(max_month)
        return df

    def compute_cac_payback(self, cac: float, monthly_churn: float,
                            arpu: float, gross_margin: float = 0.72) -> float:
        """Computes the CAC payback period in months for a given cohort configuration."""
        monthly_gp = arpu * gross_margin
        survival = SurvivalCurve(monthly_churn, self.n_months, "modified_bg").compute()
        cumulative = 0.0
        for t, s in enumerate(survival):
            cumulative += s * monthly_gp
            if cumulative >= cac:
                return float(t + 1)
        return float("inf")