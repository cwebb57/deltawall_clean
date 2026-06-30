"""
app.py — Flask server for deltawall_clean
Serves GEX chart data and lottery contract picks for SPX / QQQ
"""

from __future__ import annotations

import os
from flask import Flask, jsonify, render_template
from dotenv import load_dotenv

from gex_core_mvp import (
    compute_gex,
    find_flip_strike,
    get_gamma_flip_bias,
    get_spot,
    label_top_levels,
)
from contract_selector import select_contracts

load_dotenv()

app = Flask(__name__)

TICKERS = ["SPX", "QQQ"]


# ── Health check ────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ── GEX data endpoint ───────────────────────────────────────────────────────
@app.route("/api/gex/<ticker>")
def gex_data(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        return jsonify({"error": f"Unsupported ticker: {ticker}"}), 400
    try:
        spot = get_spot(ticker)
        gex_list, spot = compute_gex(ticker, spot)
        flip = find_flip_strike(sorted(gex_list, key=lambda x: x["strike"]))
        bias = get_gamma_flip_bias(spot, flip)
        levels = label_top_levels(gex_list, spot, flip)

        # Filter to strikes within 3% of spot for chart clarity
        band = 0.03
        filtered = [g for g in gex_list
                    if spot * (1 - band) <= g["strike"] <= spot * (1 + band)]

        return jsonify({
            "ticker": ticker,
            "spot": round(spot, 2),
            "flip": flip,
            "bias": bias,
            "levels": levels,
            "gex_bars": filtered,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Lottery contract endpoint ───────────────────────────────────────────────
@app.route("/api/contracts/<ticker>")
def contracts(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        return jsonify({"error": f"Unsupported ticker: {ticker}"}), 400
    try:
        result = select_contracts(ticker, top_n=8)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Dashboard ───────────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template("gex_chart_SPX_QQQ_full.html")


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
