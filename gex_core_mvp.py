"""
gex_core_mvp.py — Core GEX engine for deltawall_clean.
Provides: compute_gex, find_flip_strike, get_gamma_flip_bias,
          get_spot, label_top_levels, theta_chain
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Any

import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ── ThetaData credentials ───────────────────────────────────────────────────
THETA_USERNAME   = os.getenv("THETA_USERNAME") or os.getenv("THETADATA_USERNAME", "")
THETA_PASSWORD   = os.getenv("THETA_PASSWORD") or os.getenv("THETADATA_PASSWORD", "")
THETADATA_API_KEY = os.getenv("THETADATA_API_KEY", "")
THETA_TIMEOUT    = int(os.getenv("THETA_TIMEOUT", "30"))
THETA_MAX_RETRIES = int(os.getenv("THETA_MAX_RETRIES", "3"))

THETA_BASE = "http://127.0.0.1:25510"   # ThetaData terminal REST endpoint

# ── Ticker normalisation ────────────────────────────────────────────────────
_YF_MAP = {"SPX": "^GSPC", "QQQ": "QQQ", "SPY": "SPY"}

def _yf_ticker(ticker: str) -> str:
    return _YF_MAP.get(ticker.upper(), ticker.upper())


# ── Spot price ──────────────────────────────────────────────────────────────
def get_spot(ticker: str) -> float:
    """Return current spot price via yfinance."""
    t = yf.Ticker(_yf_ticker(ticker))
    info = t.fast_info
    price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
    if price:
        return float(price)
    hist = t.history(period="1d", interval="1m")
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    raise ValueError(f"Cannot fetch spot for {ticker}")


# ── ThetaData options chain ─────────────────────────────────────────────────
def _theta_headers() -> dict:
    h = {"Accept": "application/json"}
    if THETADATA_API_KEY:
        h["X-API-Key"] = THETADATA_API_KEY
    elif THETA_USERNAME and THETA_PASSWORD:
        import base64
        creds = base64.b64encode(f"{THETA_USERNAME}:{THETA_PASSWORD}".encode()).decode()
        h["Authorization"] = f"Basic {creds}"
    return h


def theta_chain(ticker: str, limit: int = 800) -> list[dict]:
    """
    Fetch 0DTE options chain from ThetaData REST terminal.
    Returns list of contract dicts with greeks, quote, OI.
    Falls back to empty list on any error so the app degrades gracefully.
    """
    today = date.today().isoformat().replace("-", "")
    # SPX uses SPXW for 0DTE weeklies
    root = "SPXW" if ticker.upper() == "SPX" else ticker.upper()

    url = f"{THETA_BASE}/v2/bulk_snapshot/option/quote"
    params = {
        "root": root,
        "exp": today,
        "limit": limit,
    }

    for attempt in range(THETA_MAX_RETRIES):
        try:
            r = requests.get(url, params=params, headers=_theta_headers(),
                             timeout=THETA_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            contracts = _parse_theta_response(data)
            if contracts:
                return contracts
        except Exception as e:
            print(f"[theta_chain] attempt {attempt+1} failed: {e}")
            time.sleep(1.5)

    print(f"[theta_chain] all retries exhausted for {ticker} — returning empty chain")
    return []


def _parse_theta_response(data: dict) -> list[dict]:
    """Parse ThetaData bulk snapshot response into contract list."""
    out = []
    responses = data.get("response", [])
    for item in responses:
        try:
            contract = item.get("contract", {})
            greeks   = item.get("greeks", {})
            quote    = item.get("quote", {})
            oi_data  = item.get("open_interest", {})
            volume   = item.get("trade", {})

            strike = float(contract.get("strike", 0)) / 1000  # ThetaData stores strike * 1000
            cp     = contract.get("right", "C")
            expiry = contract.get("expiration", "")

            out.append({
                "details": {
                    "strike_price": strike,
                    "contract_type": "call" if cp == "C" else "put",
                    "expiration_date": expiry,
                },
                "greeks": {
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "theta": greeks.get("theta"),
                    "vega":  greeks.get("vega"),
                    "iv":    greeks.get("iv"),
                },
                "last_quote": {
                    "bid": float(quote.get("bid", 0) or 0),
                    "ask": float(quote.get("ask", 0) or 0),
                },
                "open_interest": int(oi_data.get("open_interest", 0) or 0),
                "day": {
                    "volume": int(volume.get("daily_volume", 0) or 0),
                },
            })
        except Exception:
            continue
    return out


# ── GEX computation ─────────────────────────────────────────────────────────
def compute_gex(ticker: str, spot: float) -> tuple[list[dict], float]:
    """
    Compute GEX per strike from options chain.
    GEX = gamma * OI * spot^2 * 0.01 * contract_multiplier
    Returns (gex_list, spot).
    """
    chain = theta_chain(ticker)
    multiplier = 100  # standard equity/index multiplier

    strike_map: dict[float, float] = {}
    for c in chain:
        try:
            strike = float(c["details"]["strike_price"])
            gamma  = float(c["greeks"].get("gamma") or 0)
            oi     = int(c.get("open_interest", 0) or 0)
            cp     = c["details"]["contract_type"].upper()

            # Calls add positive GEX, puts subtract (dealer perspective)
            sign = 1 if "CALL" in cp or cp == "C" else -1
            gex  = sign * gamma * oi * (spot ** 2) * 0.01 * multiplier
            strike_map[strike] = strike_map.get(strike, 0) + gex
        except Exception:
            continue

    gex_list = [{"strike": k, "gex": v} for k, v in sorted(strike_map.items())]
    return gex_list, spot


# ── Gamma flip ──────────────────────────────────────────────────────────────
def find_flip_strike(sorted_gex: list[dict]) -> float | None:
    """
    Find the gamma flip strike — where cumulative GEX crosses zero.
    sorted_gex must be sorted ascending by strike.
    """
    cumulative = 0.0
    prev_strike = None
    for item in sorted_gex:
        prev = cumulative
        cumulative += item["gex"]
        if prev_strike is not None and prev < 0 <= cumulative:
            return item["strike"]
        if prev_strike is not None and prev > 0 >= cumulative:
            return item["strike"]
        prev_strike = item["strike"]
    return None


def get_gamma_flip_bias(spot: float, flip: float | None) -> dict:
    """Return bias dict based on spot vs gamma flip."""
    if flip is None:
        return {"bias": "NEUTRAL", "flip": None, "distance_pct": None}
    dist_pct = (spot - flip) / flip * 100
    bias = "BULLISH" if spot > flip else "BEARISH"
    return {"bias": bias, "flip": flip, "distance_pct": round(dist_pct, 2)}


# ── Key level labeling ──────────────────────────────────────────────────────
def label_top_levels(gex_list: list[dict], spot: float,
                     flip: float | None, top_n: int = 5) -> list[dict]:
    """
    Label the top GEX levels as Call Wall, Put Wall, or Key Support/Resistance.
    Call Wall = largest positive GEX above spot
    Put Wall  = largest negative GEX below spot (most negative)
    """
    if not gex_list:
        return []

    above = [g for g in gex_list if g["strike"] > spot and g["gex"] > 0]
    below = [g for g in gex_list if g["strike"] < spot and g["gex"] < 0]
    top_pos = sorted(gex_list, key=lambda x: x["gex"], reverse=True)[:top_n]
    top_neg = sorted(gex_list, key=lambda x: x["gex"])[:top_n]

    labeled = []
    call_wall_strike = max(above, key=lambda x: x["gex"])["strike"] if above else None
    put_wall_strike  = min(below, key=lambda x: x["gex"])["strike"] if below else None

    seen = set()
    for g in top_pos + top_neg:
        if g["strike"] in seen:
            continue
        seen.add(g["strike"])
        if g["strike"] == call_wall_strike:
            label = "Call Wall"
        elif g["strike"] == put_wall_strike:
            label = "Put Wall"
        elif g["strike"] == flip:
            label = "Gamma Flip"
        elif g["gex"] > 0:
            label = "Resistance"
        else:
            label = "Support"
        labeled.append({"strike": g["strike"], "gex": g["gex"], "type": label})

    return sorted(labeled, key=lambda x: x["strike"])
