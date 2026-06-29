"""
contract_selector.py — Score 0DTE contracts using GEX levels, liquidity, and greeks.

Bullish bias strongly favors OTM lottery calls ($0.50–$2.00, Δ 0.15–0.45, volume).
Deep ITM contracts are penalized.
"""

from __future__ import annotations

import argparse
from datetime import date
from typing import Any

from gex_core_mvp import (
    compute_gex,
    find_flip_strike,
    get_gamma_flip_bias,
    get_spot,
    label_top_levels,
    theta_chain,
)

MIN_SCORE = 25
FLIP_VERY_CLOSE_PCT = 0.002
LOTTO_DELTA_LO, LOTTO_DELTA_HI = 0.15, 0.45
LOTTO_PRICE_LO, LOTTO_PRICE_HI = 0.50, 2.00
DEEP_ITM_DELTA, VERY_DEEP_ITM_DELTA = 0.55, 0.70


def _mid(bid: float, ask: float) -> float:
    return (bid + ask) / 2 if bid > 0 and ask > 0 else (ask or bid or 0.0)


def _spr_pct(bid: float, ask: float) -> float | None:
    m = _mid(bid, ask)
    return (ask - bid) / m * 100 if m > 0 and ask > bid else None


def _liquidity_score(oi: int, bid: float, ask: float) -> tuple[int, dict[str, int]]:
    pts, bd = 0, {}
    if oi >= 2000: pts, bd["oi"] = pts + 15, 15
    elif oi >= 500: pts, bd["oi"] = pts + 10, 10
    elif oi >= 100: pts, bd["oi"] = pts + 5, 5
    sp = _spr_pct(bid, ask)
    if sp is not None:
        if sp <= 5: pts, bd["spread"] = pts + 15, 15
        elif sp <= 10: pts, bd["spread"] = pts + 10, 10
        elif sp <= 20: pts, bd["spread"] = pts + 5, 5
        elif sp > 30: pts, bd["wide_spread"] = pts - 8, -8
    elif bid > 0 or ask > 0:
        pts, bd["quote"] = pts + 3, 3
    return pts, bd


def _volume_score(volume: int, oi: int) -> tuple[int, dict[str, int]]:
    pts, bd = 0, {}
    if volume >= 5000: pts, bd["vol_hot"] = 18, 18
    elif volume >= 2000: pts, bd["vol_high"] = 14, 14
    elif volume >= 500: pts, bd["vol_good"] = 10, 10
    elif volume >= 100: pts, bd["vol_ok"] = 5, 5
    elif volume == 0 and oi >= 2000: pts, bd["oi_proxy"] = 6, 6
    elif volume == 0 and oi >= 500: pts, bd["oi_proxy"] = 3, 3
    return pts, bd


def _lottery_score(c: dict, spot: float, bias: str) -> tuple[int, dict[str, int]]:
    pts, bd = 0, {}
    side, strike = c["side"].upper(), float(c["strike"])
    delta, mid = abs(float(c.get("delta", 0) or 0)), _mid(c["bid"], c["ask"])

    in_price_band = LOTTO_PRICE_LO <= mid <= LOTTO_PRICE_HI
    in_delta_band = LOTTO_DELTA_LO <= delta <= LOTTO_DELTA_HI

    if delta >= VERY_DEEP_ITM_DELTA:
        pts, bd["deep_itm"] = pts - 28, -28
    elif delta >= DEEP_ITM_DELTA and not in_price_band:
        pts, bd["itm_heavy"] = pts - 14, -14
    elif delta >= 0.50 and not in_delta_band and not in_price_band:
        pts, bd["itm_mild"] = pts - 6, -6
    elif in_price_band and 0.45 <= delta <= 0.55:
        pts, bd["near_atm_lotto"] = pts + 14, 14

    if in_price_band:
        pts, bd["lotto_price"] = pts + 16, 16
    elif 0.20 <= mid < LOTTO_PRICE_LO and in_delta_band:
        pts, bd["cheap_otm_lotto"] = pts + 12, 12
    elif 0.30 <= mid < LOTTO_PRICE_LO:
        pts, bd["cheap_lotto"] = pts + 4, 4
    elif LOTTO_PRICE_HI < mid <= 3.00: pts, bd["price_ok"] = pts + 4, 4
    elif mid > 5.00: pts, bd["too_expensive"] = pts - 12, -12
    elif mid < 0.40: pts, bd["penny"] = pts - 18, -18

    if LOTTO_DELTA_LO <= delta <= LOTTO_DELTA_HI: pts, bd["lotto_delta"] = pts + 14, 14
    elif delta < 0.08: pts, bd["too_wing"] = pts - 12, -12
    elif 0.10 <= delta < LOTTO_DELTA_LO: pts, bd["wing_delta"] = pts + 4, 4
    elif LOTTO_DELTA_HI < delta < DEEP_ITM_DELTA: pts, bd["high_delta"] = pts - 4, -4

    if bias == "BULLISH" and side == "CALL":
        if strike > spot * 1.003: pts, bd["otm_call"] = pts + 12, 12
        if strike > spot and LOTTO_DELTA_LO <= delta <= LOTTO_DELTA_HI and LOTTO_PRICE_LO <= mid <= LOTTO_PRICE_HI:
            pts, bd["bull_lotto_sweet"] = pts + 20, 20
    elif bias == "BEARISH" and side == "PUT":
        if strike < spot * 0.997: pts, bd["otm_put"] = pts + 12, 12
        if strike < spot and LOTTO_DELTA_LO <= delta <= LOTTO_DELTA_HI and LOTTO_PRICE_LO <= mid <= LOTTO_PRICE_HI:
            pts, bd["bear_lotto_sweet"] = pts + 20, 20
    elif (bias == "BULLISH" and side == "PUT") or (bias == "BEARISH" and side == "CALL"):
        pts, bd["fights_bias"] = pts - 10, -10
    return pts, bd


def _greeks_score(delta: float, gamma: float, theta: float) -> tuple[int, dict[str, int]]:
    pts, bd = 0, {}
    if gamma > 0: pts, bd["gamma"] = pts + 4, 4
    if theta > -2.0: pts, bd["theta"] = pts + 4, 4
    elif theta > -5.0: pts, bd["theta"] = pts + 2, 2
    return pts, bd


def _gex_score(spot, flip, strike, side, call_wall, put_wall) -> tuple[int, dict[str, int]]:
    pts, bd = 0, {}
    is_call = side.upper() == "CALL"
    bias = get_gamma_flip_bias(spot, flip)["bias"]
    if flip:
        if bias == "BULLISH" and spot > flip and is_call: pts, bd["flip_side"] = 12, 12
        elif bias == "BEARISH" and spot < flip and not is_call: pts, bd["flip_side"] = 12, 12
    if is_call and call_wall and spot * 0.995 <= strike <= call_wall: pts, bd["call_wall"] = 6, 6
    if not is_call and put_wall and put_wall <= strike <= spot * 1.005: pts, bd["put_wall"] = 6, 6
    if (bias == "BULLISH" and not is_call) or (bias == "BEARISH" and is_call): pts, bd["fights_gex"] = pts - 12, -12
    return max(0, pts), bd


def fetch_contracts_near_spot(ticker: str, spot: float, bias: str = "NEUTRAL", pct_band: float = 0.03) -> list[dict]:
    call_band = 0.08 if bias == "BULLISH" else pct_band
    put_band = 0.08 if bias == "BEARISH" else pct_band
    out = []
    for c in theta_chain(ticker, limit=800):
        strike = float(c.get("details", {}).get("strike_price", 0) or 0)
        if strike <= 0 or spot <= 0: continue
        cp = c.get("details", {}).get("contract_type", "call").upper()
        is_call = "CALL" in cp or cp == "C"
        dist = (strike - spot) / spot
        if is_call and (dist < -pct_band or dist > call_band): continue
        if not is_call and (dist > pct_band or dist < -put_band): continue
        g = c.get("greeks", {})
        bid = float(c.get("last_quote", {}).get("bid", 0) or 0)
        ask = float(c.get("last_quote", {}).get("ask", 0) or 0)
        out.append({
            "strike": strike, "side": "CALL" if is_call else "PUT",
            "delta": float(g.get("delta", 0) or 0), "gamma": float(g.get("gamma", 0) or 0),
            "theta": float(g.get("theta", 0) or 0), "oi": int(c.get("open_interest", 0) or 0),
            "volume": int((c.get("day") or {}).get("volume", 0) or 0),
            "bid": bid, "ask": ask, "expiry": date.today().isoformat(),
        })
    return out


def score_contract(c, spot, flip, gex_list, call_wall, put_wall, bias) -> dict[str, Any]:
    gex_pts, gex_bd = _gex_score(spot, flip, c["strike"], c["side"], call_wall, put_wall)
    liq_pts, liq_bd = _liquidity_score(c["oi"], c["bid"], c["ask"])
    vol_pts, vol_bd = _volume_score(c.get("volume", 0), c["oi"])
    lotto_pts, lotto_bd = _lottery_score(c, spot, bias)
    greek_pts, greek_bd = _greeks_score(c["delta"], c["gamma"], c["theta"])
    mid = _mid(c["bid"], c["ask"])
    sp = _spr_pct(c["bid"], c["ask"])
    return {**c, "score": gex_pts + liq_pts + vol_pts + lotto_pts + greek_pts,
            "gex_pts": gex_pts, "liq_pts": liq_pts, "vol_pts": vol_pts,
            "lotto_pts": lotto_pts, "greek_pts": greek_pts,
            "mid": round(mid, 2) if mid else None,
            "spread_pct": round(sp, 1) if sp is not None else None,
            "stop": round(flip * 0.9985, 2) if flip and c["side"] == "CALL" else (round(flip * 1.0015, 2) if flip else None),
            "breakdown": {**gex_bd, **liq_bd, **vol_bd, **lotto_bd, **greek_bd}}


def _rank_key(p: dict) -> tuple:
    mid, delta = p.get("mid") or 0, abs(float(p.get("delta", 0) or 0))
    sweet = int(LOTTO_PRICE_LO <= mid <= LOTTO_PRICE_HI and LOTTO_DELTA_LO <= delta <= LOTTO_DELTA_HI)
    return (sweet, p["score"], p.get("volume") or 0)


def select_contracts(ticker: str, top_n: int = 8) -> dict[str, Any]:
    spot = get_spot(ticker)
    gex_list, spot = compute_gex(ticker, spot)
    flip = find_flip_strike(sorted(gex_list, key=lambda x: x["strike"]))
    bias_info = get_gamma_flip_bias(spot, flip)
    bias = bias_info["bias"]
    labeled = label_top_levels(gex_list, spot, flip)
    call_wall = next((lv["strike"] for lv in labeled if lv["type"] == "Call Wall"), None)
    put_wall = next((lv["strike"] for lv in labeled if lv["type"] == "Put Wall"), None)
    scored = [score_contract(c, spot, flip, gex_list, call_wall, put_wall, bias)
              for c in fetch_contracts_near_spot(ticker, spot, bias)]
    if bias == "BULLISH":
        scored = [s for s in scored if s["side"] == "CALL"] or scored
    elif bias == "BEARISH":
        scored = [s for s in scored if s["side"] == "PUT"] or scored
    scored.sort(key=lambda x: (-_rank_key(x)[0], -_rank_key(x)[1], -_rank_key(x)[2]))
    picks = [s for s in scored if s["score"] >= MIN_SCORE and s["bid"] > 0 and s["ask"] > 0][:top_n]
    return {"ticker": ticker.upper(), "spot": spot, "flip": flip, "call_wall": call_wall,
            "put_wall": put_wall, "bias": bias_info, "picks": picks, "scored_count": len(scored)}


def print_report(result: dict[str, Any]) -> None:
    print(f"\n=== 0DTE Contract Selector: {result['ticker']} ===")
    print(f"Spot: {result['spot']:,.2f}  Flip: {result['flip']}  Bias: {result['bias']['bias']}")
    for i, p in enumerate(result["picks"], 1):
        print(f"  {i}. {p['side']} {p['strike']:.0f}  score={p['score']}  LOT={p['lotto_pts']}  "
              f"Δ={p['delta']:.2f}  mid=${p.get('mid')}  OI={p['oi']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="SPX")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    print_report(select_contracts(a.ticker.upper(), top_n=a.top))


if __name__ == "__main__":
    main()