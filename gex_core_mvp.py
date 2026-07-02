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
                "reason": "Within +/-0.3% of Gamma Flip - caution"}
    return {"bias": "BULLISH" if dist_pct > 0 else "BEARISH", "dist_pct": dist_pct,
            "caution": False, "reason": "Standard flip bias"}


def label_top_levels(gex_list, spot=0, flip=None, n=5):
    """
    Identify Call Wall (largest positive GEX strike above spot) and
    Put Wall (largest negative GEX strike below spot), plus top |GEX|
    levels overall. Returns list of dicts: {"strike", "gex", "type"}.
    Called as label_top_levels(gex_list, spot, flip) by contract_selector.py.
    """
    if not gex_list:
        return []

    out = []

    above = [r for r in gex_list if spot and r.get("strike", 0) > spot and r.get("gex", 0) > 0]
    below = [r for r in gex_list if spot and r.get("strike", 0) < spot and r.get("gex", 0) < 0]

    call_wall = max(above, key=lambda r: r.get("gex", 0)) if above else None
    put_wall = min(below, key=lambda r: r.get("gex", 0)) if below else None

    if call_wall:
        out.append({"strike": call_wall["strike"], "gex": call_wall["gex"], "type": "Call Wall"})
    if put_wall:
        out.append({"strike": put_wall["strike"], "gex": put_wall["gex"], "type": "Put Wall"})

    if flip:
        out.append({"strike": flip, "gex": None, "type": "Gamma Flip"})

    used_strikes = {r["strike"] for r in out}
    ranked = sorted(
        (r for r in gex_list if r.get("strike") not in used_strikes),
        key=lambda r: abs(r.get("gex", 0)), reverse=True
    )
    for r in ranked[:max(0, n - len(out))]:
        out.append({"strike": r["strike"], "gex": r.get("gex"), "type": "Level"})

    return out

