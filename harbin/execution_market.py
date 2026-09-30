from __future__ import annotations

import json

from .market_intel import MarketIntelligence as _BaseMarketIntelligence


class MarketIntelligence(_BaseMarketIntelligence):
    """Stage 7 execution wrapper that exposes the timestamp of every shopped quote."""

    @staticmethod
    def _summary(g, quotes):
        out = _BaseMarketIntelligence._summary(g, quotes)
        try:
            audit = json.loads(out.get("market_quotes_json") or "[]")
        except Exception:
            audit = []
        by_provider = {str(q.get("provider") or ""): q for q in audit if isinstance(q, dict)}
        mapping = {
            "best_home_ml_quote_at": "best_home_ml_book",
            "best_away_ml_quote_at": "best_away_ml_book",
            "best_home_spread_quote_at": "best_home_spread_book",
            "best_away_spread_quote_at": "best_away_spread_book",
            "best_over_quote_at": "best_over_book",
            "best_under_quote_at": "best_under_book",
        }
        for target, book_field in mapping.items():
            book = str(out.get(book_field) or "")
            quote = by_provider.get(book) or {}
            out[target] = quote.get("last_update")
        return out
