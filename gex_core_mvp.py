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

THETA_DIRECT = os.getenv("THETA_DIRECT_URL", "http://127.0.0.1:25510").rstrip("/")
THETA_PROXY  = os.getenv("RAILWAY_PROXY_URL", "https://web-production-e90c.up.railway.app").rstrip("/")

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


def _to_theta_rows(obj: Any) -> list[dict]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [{str(k).lower(): v for k, v in (r.items() if isinstance(r, dict) else {})} for r in obj]
    if hasattr(obj, "to_dicts"):
        return [{str(k).lower(): v for k, v in r.items()} for r in obj.to_dicts()]
    if hasattr(obj, "iterrows"):
        return [{str(k).lower(): v for k, v in dict(row).items()} for _, row in obj.iterrows()]
    if hasattr(obj, "to_dict"):
        raw = obj.to_dict()
        if isinstance(raw, list):
            return [{str(k).lower(): v for k, v in (r.items() if isinstance(r, dict) else {})} for r in raw]
        if isinstance(raw, dict):
            return [{str(k).lower(): v for k, v in raw.items()}]
    return []


def _fetch_theta_chain(base: str, ticker: str, limit: int) -> list[dict]:
    """Fetch 0DTE chain from a ThetaData REST base (local terminal or HTTP proxy)."""
    today = date.today().isoformat().replace("-", "")
    root = "SPXW" if ticker.upper() == "SPX" else ticker.upper()
    url = f"{base.rstrip('/')}/v2/bulk_snapshot/option/quote"
    params = {"root": root, "exp": today, "limit": limit}
    r = requests.get(url, params=params, headers=_theta_headers(), timeout=THETA_TIMEOUT)
    r.raise_for_status()
    return _parse_theta_response(r.json())


def _theta_chain_via_client(ticker: str) -> list[dict]:
    """Cloud ThetaData client fallback — same path used by option_trader_proxy on Railway."""
    try:
        from thetadata import ThetaClient
    except ImportError:
        print("[theta_chain] thetadata package not installed")
        return []
    if not THETADATA_API_KEY:
        print("[theta_chain] THETADATA_API_KEY not set — cloud fallback unavailable")
        return []

    sym = "SPXW" if ticker.upper() in ("SPX", "SPXW", "I:SPX") else ticker.upper()
    exp = date.today()
    client = ThetaClient(api_key=THETADATA_API_KEY)

    g_rows = _to_theta_rows(client.option_snapshot_greeks_first_order(symbol=sym, expiration=exp))
    try:
        oi_rows = _to_theta_rows(
            client.option_snapshot_open_interest(symbol=sym, expiration=exp, strike="*", right="both")
        )
    except Exception as e:
        print(f"[theta_chain] OI fetch note: {e}")
        oi_rows = []
    try:
        g2_rows = _to_theta_rows(client.option_snapshot_greeks_second_order(symbol=sym, expiration=exp))
    except Exception as e:
        print(f"[theta_chain] gamma second order note: {e}")
        g2_rows = []

    gamma_lookup = {
        (round(float(r.get("strike", 0) or 0), 2), str(r.get("right", "") or "")[:1].upper()): r.get("gamma")
        for r in g2_rows if r.get("gamma") is not None
    }
    oicol = next((c for r in oi_rows[:1] for c in r if "open" in c or c == "oi"), "open_interest")
    oi_lookup = {
        (round(float(r.get("strike", 0) or 0), 2), str(r.get("right", "") or "")[:1].upper()):
        r.get(oicol, r.get("open_interest"))
        for r in oi_rows
    }

    contracts: list[dict] = []
    for r in g_rows:
        try:
            strike = float(r.get("strike", r.get("strike_price", 0)) or 0)
            rt = str(r.get("right", r.get("call_put", "C")) or "C")[0].upper()
            if rt not in ("C", "P") or strike <= 0:
                continue
            key = (round(strike, 2), rt)
            contracts.append({
                "details": {
                    "strike_price": strike,
                    "contract_type": "call" if rt == "C" else "put",
                    "expiration_date": exp.isoformat(),
                },
                "greeks": {
                    "delta": float(r.get("delta", 0) or 0),
                    "gamma": float(r.get("gamma") or gamma_lookup.get(key) or 0),
                    "theta": float(r.get("theta", 0) or 0),
                    "vega": float(r.get("vega", 0) or 0),
                    "iv": float(r.get("implied_volatility", r.get("iv", 0)) or 0),
                },
                "last_quote": {
                    "bid": float(r.get("bid", 0) or 0),
                    "ask": float(r.get("ask", 0) or 0),
                },
                "open_interest": int(r.get("open_interest") or oi_lookup.get(key) or 0),
                "day": {"volume": 0},
            })
        except Exception:
            continue
    return contracts


def theta_chain(ticker: str, limit: int = 800) -> list[dict]:
    """
    Fetch 0DTE options chain — local terminal first, HTTP proxy, then cloud client.
    Returns list of contract dicts with greeks, quote, OI.
    """
    sources: list[tuple[str, str]] = [("direct", THETA_DIRECT)]
    if THETA_PROXY and THETA_PROXY != THETA_DIRECT:
        sources.append(("proxy", THETA_PROXY))

    for source_name, base in sources:
        for attempt in range(THETA_MAX_RETRIES):
            try:
                contracts = _fetch_theta_chain(base, ticker, limit)
                if contracts:
                    print(f"[theta_chain] {ticker}: {len(contracts)} contracts via {source_name} ({base})")
                    return contracts
                print(f"[theta_chain] {source_name} returned empty chain for {ticker}")
                break
            except Exception as e:
                print(f"[theta_chain] {source_name} attempt {attempt + 1} failed: {e}")
                time.sleep(1.5)

    try:
        contracts = _theta_chain_via_client(ticker)
        if contracts:
            print(f"[theta_chain] {ticker}: {len(contracts)} contracts via cloud client")
            return contracts
    except Exception as e:
        print(f"[theta_chain] cloud client failed: {e}")

    print(f"[theta_chain] all sources exhausted for {ticker} — returning empty chain")
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
