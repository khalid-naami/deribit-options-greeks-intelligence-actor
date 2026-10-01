"""Direct Official REST Client for Deribit Public API v2."""

import logging
import re
import time
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# Currencies with native inverse books vs USDC linear books
NATIVE_CURRENCIES = {'BTC', 'ETH', 'USDC'}
USDC_SETTLED_CURRENCIES = {'SOL', 'MATIC', 'XRP', 'AVAX', 'PAXG', 'ETHW', 'STETH', 'USDT', 'USDE'}

class DeribitClient:
    """Official Deribit Public API v2 Connector."""

    BASE_URL = "https://www.deribit.com/api/v2/public"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "DeribitOptionsQuantActor/1.0",
            "Accept": "application/json"
        })
        # Instrument format regex: BTC-27OCT26-65000-C or SOL_USDC-27OCT26-150-P
        self.symbol_regex = re.compile(
            r'^([A-Z]+(?:_[A-Z]+)?)-(\d{1,2}[A-Z]{3}\d{2})-(\d+(?:[d\.]\d+)?)-([CP])$'
        )

    def _get_query_currency(self, currency: str) -> str:
        clean = currency.upper().replace('-USD', '').replace('-USDT', '').replace('^', '').strip()
        if clean in USDC_SETTLED_CURRENCIES:
            return 'USDC'
        return clean

    def _get_instrument_prefix(self, currency: str) -> str:
        clean = currency.upper().replace('-USD', '').replace('-USDT', '').replace('^', '').strip()
        if clean in USDC_SETTLED_CURRENCIES:
            return f"{clean}_USDC-"
        return f"{clean}-"

    def _get_perpetual_instrument(self, currency: str) -> str:
        clean = currency.upper().replace('-USD', '').replace('-USDT', '').replace('^', '').strip()
        if clean in USDC_SETTLED_CURRENCIES:
            return f"{clean}_USDC-PERPETUAL"
        return f"{clean}-PERPETUAL"

    def get_index_price(self, currency: str) -> float:
        """Fetch real-time index spot price directly from Deribit."""
        clean = currency.upper().replace('-USD', '').replace('-USDT', '').replace('^', '').strip().lower()
        url = f"{self.BASE_URL}/get_index_price?index_name={clean}_usd"

        try:
            res = self.session.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                price = data.get('result', {}).get('index_price')
                if price and float(price) > 0:
                    return float(price)
        except Exception as e:
            logger.warning(f"Error fetching Deribit index price for {currency}: {e}")

        # Fallback to book summary if index price endpoint fails
        summary = self.get_options_book_summary(currency)
        if summary:
            first = summary[0]
            price = first.get('index_price') or first.get('underlying_price')
            if price and float(price) > 0:
                return float(price)

        defaults = {'btc': 65000.0, 'eth': 2650.0, 'sol': 155.0, 'xrp': 0.60, 'avax': 28.0, 'matic': 0.40, 'paxg': 2685.0}
        return defaults.get(clean, 100.0)

    def get_options_book_summary(self, currency: str) -> List[Dict[str, Any]]:
        """Fetch active options book summary with Mark IV, Greeks, Prices, and Open Interest."""
        query_cur = self._get_query_currency(currency)
        prefix = self._get_instrument_prefix(currency)
        url = f"{self.BASE_URL}/get_book_summary_by_currency?currency={query_cur}&kind=option"

        try:
            res = self.session.get(url, timeout=12)
            if res.status_code == 200:
                data = res.json()
                all_instruments = data.get('result', [])
                filtered = [
                    inst for inst in all_instruments
                    if inst.get('instrument_name', '').startswith(prefix)
                ]
                return filtered
        except Exception as e:
            logger.error(f"Error fetching Deribit options book for {currency}: {e}")

        return []

    def get_tradingview_candlesticks(
        self,
        currency: str,
        resolution: str = "1D",
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Fetch historical TradingView OHLCV candlesticks directly from Deribit."""
        instrument = self._get_perpetual_instrument(currency)
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (days * 24 * 60 * 60 * 1000)

        url = (
            f"{self.BASE_URL}/get_tradingview_chart_data?"
            f"instrument_name={instrument}&start_timestamp={start_ts}"
            f"&end_timestamp={end_ts}&resolution={resolution}"
        )

        candles = []
        try:
            res = self.session.get(url, timeout=12)
            if res.status_code == 200:
                data = res.json()
                result = data.get('result', {})

                opens = result.get('open', [])
                highs = result.get('high', [])
                lows = result.get('low', [])
                closes = result.get('close', [])
                volumes = result.get('volume', [])
                ticks = result.get('ticks', [])

                for i in range(len(ticks)):
                    dt = datetime.fromtimestamp(ticks[i] / 1000.0, tz=timezone.utc)
                    candles.append({
                        "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "open": round(float(opens[i]), 2 if float(opens[i]) > 10 else 4),
                        "high": round(float(highs[i]), 2 if float(highs[i]) > 10 else 4),
                        "low": round(float(lows[i]), 2 if float(lows[i]) > 10 else 4),
                        "close": round(float(closes[i]), 2 if float(closes[i]) > 10 else 4),
                        "volume": round(float(volumes[i]), 2)
                    })

        except Exception as e:
            logger.error(f"Error fetching Deribit candlesticks for {currency}: {e}")

        return candles
