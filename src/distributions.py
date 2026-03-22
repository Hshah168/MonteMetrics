import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ChurnDistribution:
    alpha: float = 2.0
    beta: float = 38.0
    recession_multiplier: float = 1.4

    def sample(self, n: int, recession_flags: Optional[np.ndarray] = None) -> np.ndarray:
        """Generates samples of monthly churn rates, optionally adjusting for recession scenarios."""
        base = np.random.beta(self.alpha, self.beta, size=n)
        if recession_flags is not None:
            base = base * (1 + (self.recession_multiplier - 1) * recession_flags)
        return np.clip(base, 0, 1)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @classmethod
    def from_historical(cls, observed_churn_rates: List[float]) -> "ChurnDistribution":
        arr = np.array(observed_churn_rates)
        mu = arr.mean()
        var = max(arr.var(), 1e-6)
        factor = mu * (1 - mu) / var - 1
        alpha = mu * factor
        beta_param = (1 - mu) * factor
        return cls(alpha=max(alpha, 0.1), beta=max(beta_param, 0.1))


@dataclass
class ARPUDistribution:
    mu: float = 4.4
    sigma: float = 0.45
    gross_margin: float = 0.72
    """Represents the distribution of ARPU, modeled as a log-normal distribution with parameters mu and sigma. The gross_margin field is used to compute net revenue from ARPU samples."""

    def sample(self, n: int) -> np.ndarray:
        return np.random.lognormal(mean=self.mu, sigma=self.sigma, size=n)

    @property
    def expected_value(self) -> float:
        return np.exp(self.mu + 0.5 * self.sigma ** 2)

    def net_revenue(self, n: int) -> np.ndarray:
        return self.sample(n) * self.gross_margin


@dataclass
class CACDistribution:
    shape: float = 6.0
    scale: float = 30.0

    def sample(self, n: int) -> np.ndarray:
        return np.random.gamma(shape=self.shape, scale=self.scale, size=n)

    @property
    def mean(self) -> float:
        return self.shape * self.scale

    @property
    def std(self) -> float:
        return np.sqrt(self.shape) * self.scale


@dataclass
class MacroShockDistribution:
    recession_prob: float = 0.12

    def sample_monthly(self, n: int, n_months: int) -> np.ndarray:
        monthly_prob = 1 - (1 - self.recession_prob) ** (1 / 12)
        return np.random.binomial(1, monthly_prob, size=(n, n_months)).astype(float)


def create_correlated_inputs(n: int, churn_arpu_correlation: float = -0.3):
    """Generates correlated samples of churn and ARPU based on a specified correlation coefficient."""
    from scipy.stats import norm, beta as beta_dist, lognorm as lognorm_dist
    corr = np.array([[1.0, churn_arpu_correlation],
                     [churn_arpu_correlation, 1.0]])
    L = np.linalg.cholesky(corr)
    z = np.random.randn(2, n)
    correlated = L @ z
    u_churn = norm.cdf(correlated[0])
    u_arpu = norm.cdf(correlated[1])
    cd = ChurnDistribution()
    ad = ARPUDistribution()
    churn = beta_dist.ppf(u_churn, cd.alpha, cd.beta)
    arpu = lognorm_dist.ppf(u_arpu, s=ad.sigma, scale=np.exp(ad.mu))
    return churn, arpu