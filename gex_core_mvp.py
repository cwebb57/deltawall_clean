"""
gex_core_mvp.py
Thin client that pulls already-computed GEX/spot/flip/regime data
from the deltawall-proxy engine (which owns the ThetaData connection).
"""

import requests

PROXY_BASE = "https://web-production-e90c.up.railway.app"


def fetch_gex_from_proxy(ticker: str) -> dict:
    """Pull already-computed GEX/spot/flip/regime from the proxy."""
    try:
        resp = requests.get(f"{PROXY_BASE}/api/stock/{ticker}/gex-computed", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[proxy] fetch error for {ticker}: {e}")
        return {
            "data": [], "spot": 0, "net_gex": 0,
            "flip_strike": None, "flip_analysis": {},
            "regime": "NEUTRAL", "regime_score": 0,
        }


def get_spot(ticker: str) -> float:
    return fetch_gex_from_proxy(ticker).get("spot", 0) or 0


def compute_gex(ticker: str, spot: float = 0):
    payload = fetch_gex_from_proxy(ticker)
    gex_list = payload.get("data", [])
    spot = payload.get("spot", spot) or spot
    return gex_list, spot


def find_flip_strike(gex_list):
    for i in range(len(gex_list) - 1):
        if gex_list[i]['gex'] * gex_list[i + 1]['gex'] < 0:
            return gex_list[i]['strike']
    return None


def get_gamma_flip_bias(spot, flip):
    if not spot or not flip:
        return {"bias": "NEUTRAL", "dist_pct": None, "caution": True,
                "reason": "No reliable Gamma Flip available"}
    dist_pct = round((spot - flip) / spot * 100, 3)
    if abs(dist_pct) <= 0.3:
        return {"bias": "NEUTRAL", "dist_pct": dist_pct, "caution": True,
                "reason": "Within ±0.3% of Gamma Flip — caution"}
    return {"bias": "BULLISH" if dist_pct > 0 else "BEARISH", "dist_pct": dist_pct,
            "caution": False, "reason": "Standard flip bias"}


def label_top_levels(gex_list, n=5):
    if not gex_list:
        return []
    return sorted(gex_list, key=lambda x: abs(x.get('gex', 0)), reverse=True)[:n]
