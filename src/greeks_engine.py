"""Quantitative Engine for Deribit Crypto Options Chains, Greeks & Volume Distribution."""

import logging
import math
from datetime import datetime, date, timezone
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy.stats import norm

from src.deribit_client import DeribitClient

logger = logging.getLogger(__name__)

class DeribitGreeksEngine:
    """Parses Deribit live option books into unified strike-level Greeks & OI/Volume records."""

    def __init__(self, client: DeribitClient):
        self.client = client

    def process_currency_options(
        self,
        currency: str,
        max_expirations: int = 4
    ) -> Dict[str, Any]:
        """Process complete live options chains across active expirations for a given cryptocurrency."""
        clean_curr = currency.upper().replace('-USD', '').replace('-USDT', '').replace('^', '').strip()
        index_price = self.client.get_index_price(clean_curr)
        raw_options = self.client.get_options_book_summary(clean_curr)

        if not raw_options:
            logger.warning(f"No option book records found for {clean_curr}")
            return {
                "currency": clean_curr,
                "indexPriceUsd": index_price,
                "expirationsCount": 0,
                "chains": []
            }

        # Group options by Expiration Date
        expirations_map: Dict[str, List[Dict[str, Any]]] = {}
        today = date.today()

        for opt in raw_options:
            instrument = opt.get('instrument_name', '')
            match = self.client.symbol_regex.search(instrument)
            if not match:
                continue

            _, date_str, strike_str, opt_type = match.groups()
            try:
                exp_date_obj = datetime.strptime(date_str, "%d%b%y").date()
                if exp_date_obj < today:
                    continue
                exp_iso = exp_date_obj.strftime("%Y-%m-%d")
            except ValueError:
                continue

            if exp_iso not in expirations_map:
                expirations_map[exp_iso] = []
            expirations_map[exp_iso].append({
                **opt,
                "_strike": float(strike_str.replace('d', '.')),
                "_type": opt_type,
                "_exp_date_obj": exp_date_obj
            })

        sorted_expirations = sorted(list(expirations_map.keys()))[:max_expirations]
        logger.info(f"Processing {len(sorted_expirations)} Deribit expirations for {clean_curr}: {sorted_expirations}")

        chains_results = []
        all_strike_records = []

        for exp_str in sorted_expirations:
            opts_for_exp = expirations_map[exp_str]
            exp_date_obj = opts_for_exp[0]["_exp_date_obj"]
            days_to_exp = max((exp_date_obj - today).days, 0.25)
            T = days_to_exp / 365.0
            r = 0.045
            sqrt_T = math.sqrt(max(T, 0.0001))
            discount = math.exp(-r * T)

            # Map options by Strike
            calls_by_strike: Dict[float, Dict[str, Any]] = {}
            puts_by_strike: Dict[float, Dict[str, Any]] = {}

            for opt in opts_for_exp:
                s = opt["_strike"]
                if opt["_type"] == 'C':
                    calls_by_strike[s] = opt
                else:
                    puts_by_strike[s] = opt

            all_strikes = sorted(list(set(calls_by_strike.keys()) | set(puts_by_strike.keys())))

            tot_call_oi, tot_put_oi = 0.0, 0.0
            tot_call_vol, tot_put_vol = 0.0, 0.0
            chain_strike_records = []

            for strike in all_strikes:
                c_opt = calls_by_strike.get(strike, {})
                p_opt = puts_by_strike.get(strike, {})

                # Deribit quotes prices in underlying coin (e.g. BTC) -> convert to USD
                c_mark_usd = float(c_opt.get('mark_price') or 0.0) * index_price
                p_mark_usd = float(p_opt.get('mark_price') or 0.0) * index_price

                c_iv = float(c_opt.get('mark_iv') or 0.0)
                p_iv = float(p_opt.get('mark_iv') or 0.0)
                avg_iv = (c_iv + p_iv) / 2.0 if (c_iv > 0 and p_iv > 0) else max(c_iv, p_iv, 50.0)
                sigma = avg_iv / 100.0

                c_oi = float(c_opt.get('open_interest') or 0.0)
                p_oi = float(p_opt.get('open_interest') or 0.0)
                c_vol = float(c_opt.get('volume') or 0.0)
                p_vol = float(p_opt.get('volume') or 0.0)

                tot_call_oi += c_oi
                tot_put_oi += p_oi
                tot_call_vol += c_vol
                tot_put_vol += p_vol

                # Delta from Deribit or analytical Black-Scholes
                c_delta = float(c_opt.get('delta') or 0.0)
                p_delta = float(p_opt.get('delta') or 0.0)

                # Analytical Higher-Order Greeks computation
                forward = index_price * math.exp(r * T)
                try:
                    if sigma > 0 and strike > 0 and forward > 0:
                        d1 = (math.log(forward / strike) + (0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
                        d2 = d1 - sigma * sqrt_T
                        pdf_d1 = norm.pdf(d1)
                        gamma = float((discount * pdf_d1) / (forward * sigma * sqrt_T))
                        vega = float(forward * discount * pdf_d1 * sqrt_T / 100.0)
                        theta_call = float((-(forward * discount * pdf_d1 * sigma) / (2.0 * sqrt_T) - r * c_mark_usd) / 365.0)
                        theta_put = float((-(forward * discount * pdf_d1 * sigma) / (2.0 * sqrt_T) - r * p_mark_usd) / 365.0)
                        vanna = float(-discount * pdf_d1 * (d2 / sigma))
                        charm_call = float(discount * pdf_d1 * ((r / (sigma * sqrt_T)) - (d2 / (2.0 * T))) + r * c_delta)
                        speed = float(-(gamma / forward) * ((d1 / (sigma * sqrt_T)) + 1.0))
                        vomma = float((vega * d1 * d2) / sigma)
                        color = float(-gamma * ((r / (sigma * sqrt_T)) + ((1.0 - d1 * d2) / (2.0 * T))))
                    else:
                        gamma, vega, theta_call, theta_put, vanna, charm_call, speed, vomma, color = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                except Exception:
                    gamma, vega, theta_call, theta_put, vanna, charm_call, speed, vomma, color = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

                if c_delta == 0.0 and strike > 0 and sigma > 0:
                    c_delta = float(norm.cdf(d1))
                if p_delta == 0.0 and strike > 0 and sigma > 0:
                    p_delta = float(-norm.cdf(-d1))

                dec = 2 if index_price > 10 else 4

                record = {
                    "currency": clean_curr,
                    "indexPriceUsd": round(index_price, dec),
                    "expirationDate": exp_str,
                    "daysToExpiration": days_to_exp,
                    "strike": round(strike, dec),
                    "callMarkPriceUsd": round(c_mark_usd, dec),
                    "putMarkPriceUsd": round(p_mark_usd, dec),
                    "callMarkIvPct": round(c_iv, 2),
                    "putMarkIvPct": round(p_iv, 2),
                    "callDelta": round(c_delta, 4),
                    "putDelta": round(p_delta, 4),
                    "gamma": round(gamma, 6),
                    "vega": round(vega, 4),
                    "callTheta": round(theta_call, 4),
                    "putTheta": round(theta_put, 4),
                    "vanna": round(vanna, 6),
                    "charmCall": round(charm_call, 6),
                    "speed": round(speed, 8),
                    "vomma": round(vomma, 6),
                    "color": round(color, 8),
                    "callOpenInterest": round(c_oi, 2),
                    "putOpenInterest": round(p_oi, 2),
                    "netOpenInterest": round(c_oi - p_oi, 2),
                    "callVolume24h": round(c_vol, 2),
                    "putVolume24h": round(p_vol, 2),
                    "netVolume24h": round(c_vol - p_vol, 2),
                    "callBreakevenUsd": round(strike + c_mark_usd, dec),
                    "putBreakevenUsd": round(max(strike - p_mark_usd, 0.0), dec)
                }
                chain_strike_records.append(record)
                all_strike_records.append(record)

            pcr_oi = round(tot_put_oi / tot_call_oi, 2) if tot_call_oi > 0 else 1.0
            pcr_vol = round(tot_put_vol / tot_call_vol, 2) if tot_call_vol > 0 else 1.0

            chains_results.append({
                "currency": clean_curr,
                "expirationDate": exp_str,
                "daysToExpiration": days_to_exp,
                "totalCallOpenInterest": round(tot_call_oi, 2),
                "totalPutOpenInterest": round(tot_put_oi, 2),
                "netOpenInterest": round(tot_call_oi - tot_put_oi, 2),
                "totalCallVolume24h": round(tot_call_vol, 2),
                "totalPutVolume24h": round(tot_put_vol, 2),
                "netVolume24h": round(tot_call_vol - tot_put_vol, 2),
                "putCallRatioOI": pcr_oi,
                "putCallRatioVolume": pcr_vol,
                "strikesCount": len(chain_strike_records),
                "records": chain_strike_records
            })

        return {
            "currency": clean_curr,
            "indexPriceUsd": round(index_price, 2),
            "expirationsCount": len(chains_results),
            "chains": chains_results,
            "allRecords": all_strike_records
        }
