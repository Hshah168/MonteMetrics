import yfinance as yf


def fetch_spotify_live():
    """Pull live Spotify market data via yfinance."""
    print("Fetching live Spotify data from yfinance...")
    spot = yf.Ticker("SPOT")
    info = spot.info

    live = {
        "ticker":            "SPOT",
        "company_name":      info.get("longName", "Spotify Technology S.A."),
        "current_price":     info.get("currentPrice", 0),
        "market_cap":        info.get("marketCap", 0),
        "pe_ratio":          info.get("trailingPE", 0),
        "52w_high":          info.get("fiftyTwoWeekHigh", 0),
        "52w_low":           info.get("fiftyTwoWeekLow", 0),
        "analyst_target":    info.get("targetMeanPrice", 0),
        "revenue_ttm":       info.get("totalRevenue", 0),
        "gross_margin":      info.get("grossMargins", 0),
        "operating_margin":  info.get("operatingMargins", 0),
        "cash":              info.get("totalCash", 0),
        "free_cashflow":     info.get("freeCashflow", 0),
        "ebitda":            info.get("ebitda", 0),
        "shares_outstanding":info.get("sharesOutstanding", 0),
    }

    history = spot.history(period="2y")

    print(f"  Live price      : ${live['current_price']:,.2f}")
    print(f"  Market cap      : ${live['market_cap']/1e9:.1f}B")
    print(f"  Revenue (TTM)   : ${live['revenue_ttm']/1e9:.1f}B")
    print(f"  Gross margin    : {live['gross_margin']:.1%}")
    print(f"  Cash on hand    : ${live['cash']/1e9:.1f}B")
    print(f"  Free cash flow  : ${live['free_cashflow']/1e9:.1f}B")

    return {"live": live, "history": history}


def get_spotify_subscription_metrics():
    """
    Full-year 2025 subscription metrics.
    All numbers from Spotify 2025 Annual Earnings Letter
    """
    return {
        # Subscribers (directly disclosed)
        "premium_subscribers_end_2025":  290_000_000, 
        "mau_end_2025":                  751_000_000, 
        "net_new_subs_q4_2025":            9_000_000, 
        "net_new_subs_full_year_2025":    27_000_000, 
        "avg_monthly_net_adds_2025":       2_250_000,

        # ARPU
        "monthly_arpu_eur":   4.63,
        "monthly_arpu_usd":   5.36,

        # Churn (estimated)
        "monthly_churn_estimate": 0.012,

        # Gross margin (directly disclosed) 
        "gross_margin":    0.34,

        # CAC (estimated)
        "cac_estimated_eur":  52.81,
        "cac_estimated_usd":  61.13,

        # Financials (directly disclosed)
        "full_year_revenue_2025_eur":     17_000_000_000,  
        "q4_revenue_2025_eur":             4_530_000_000,  
        "q4_premium_revenue_2025_eur":     4_013_000_000,  
        "operating_income_2025_eur":       2_200_000_000,  
        "free_cashflow_2025_eur":          2_900_000_000,  
        "cash_end_2025_eur":               9_500_000_000,  
        "operating_margin_2025":           0.13,          

        # Recession resilience 
        "recession_multiplier":  1.08,  
        "recession_prob_annual": 0.25, 
    }