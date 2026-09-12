"""Mock crypto data provider for testing.

Provides deterministic mock data for crypto assets without requiring
external API credentials. Used in test mode.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from app.core.models.market import Quote, Trade, Bar, MarketClock
from app.core.models.crypto import (
    CryptoMarketWideConditions,
    CryptoAssetCorrelation,
    CryptoTokenomics,
    TokenUnlock,
    OnChainMetrics,
    CryptoRiskType,
    CryptoRiskFactor,
    CryptoRegime,
    CryptoMarketStructure,
)
from app.services.crypto.market_provider import CryptoMarketDataProvider
from app.services.crypto.tokenomics_provider import CryptoFundamentalDataProvider
from app.services.crypto.onchain_provider import OnChainDataProvider
from app.utils.time import now_utc


class MockCryptoDataProvider(
    CryptoMarketDataProvider,
    CryptoFundamentalDataProvider,
    OnChainDataProvider
):
    """Mock provider for crypto data testing.

    Returns deterministic mock data for supported crypto symbols.
    """

    def __init__(self):
        self._supported_symbols = ["BTC", "ETH", "SOL"]
        self._base_prices = {
            "BTC": 65000.0,
            "ETH": 3500.0,
            "SOL": 150.0,
        }

    def get_quote(self, symbol: str) -> Quote:
        base = self._base_prices.get(symbol.upper(), 100.0)
        now = now_utc()
        return Quote(
            symbol=symbol.upper(),
            bid_price=base * 0.9995,
            ask_price=base * 1.0005,
            bid_size=1000,
            ask_size=1000,
            timestamp=now,
        )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        return {s: self.get_quote(s) for s in symbols}

    def get_latest_trade(self, symbol: str) -> Trade:
        base = self._base_prices.get(symbol.upper(), 100.0)
        now = now_utc()
        return Trade(
            symbol=symbol.upper(),
            price=base,
            size=10,
            timestamp=now,
        )

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Bar]:
        base = self._base_prices.get(symbol.upper(), 100.0)
        end_time = end or now_utc()
        bars = []
        for i in range(limit):
            bar_time = end_time - timedelta(hours=i)
            # Simple deterministic price movement
            price = base * (1.0 + 0.001 * (i % 20 - 10))
            bars.append(Bar(
                symbol=symbol.upper(),
                timestamp=bar_time,
                open=price * 0.999,
                high=price * 1.002,
                low=price * 0.998,
                close=price,
                volume=1000000,
            ))
        return list(reversed(bars))

    def get_market_clock(self) -> MarketClock:
        now = now_utc()
        return MarketClock(
            timestamp=now,
            is_open=True,  # Crypto is 24/7
            next_open=now + timedelta(hours=1),
            next_close=now + timedelta(hours=24),
        )

    def get_market_status(self) -> bool:
        return True  # Crypto markets are always open

    def get_market_wide_conditions(self) -> CryptoMarketWideConditions:
        return CryptoMarketWideConditions(
            total_crypto_market_cap=2.5e12,  # $2.5T
            btc_dominance=52.0,
            eth_dominance=18.0,
            stablecoin_market_cap=150e9,  # $150B
            total_market_trend="BULL_MARKET",
            fear_greed_index=65,
            funding_rates={"BTC": 0.01, "ETH": 0.015},
            open_interest=50e9,
            timestamp=now_utc(),
        )

    def get_correlation(self, asset_pair: str) -> Optional[CryptoAssetCorrelation]:
        return CryptoAssetCorrelation(
            asset_pair=asset_pair,
            correlation_30d=0.85,
            correlation_90d=0.82,
            beta=1.2,
            timestamp=now_utc(),
        )

    def get_supported_symbols(self) -> List[str]:
        return self._supported_symbols.copy()

    def get_tokenomics(self, symbol: str) -> Optional[CryptoTokenomics]:
        sym = symbol.upper()
        if sym == "BTC":
            return CryptoTokenomics(
                token_id="bitcoin",
                symbol="BTC",
                circulating_supply=19.5e6,
                maximum_supply=21e6,
                total_supply=19.5e6,
                inflation_rate=0.015,
                emission_schedule="halving every 4 years",
                market_capitalization=1.3e12,
                fully_diluted_valuation=1.4e12,
                burn_mechanism=False,
                staking_enabled=False,
                governance_token=False,
                timestamp=now_utc(),
            )
        elif sym == "ETH":
            return CryptoTokenomics(
                token_id="ethereum",
                symbol="ETH",
                circulating_supply=120e6,
                maximum_supply=None,  # No hard cap
                total_supply=120e6,
                inflation_rate=0.02,
                emission_schedule="EIP-1559 + Proof of Stake",
                market_capitalization=420e9,
                fully_diluted_valuation=None,
                burn_mechanism=True,
                staking_enabled=True,
                staking_rate=0.04,
                governance_token=True,
                timestamp=now_utc(),
            )
        elif sym == "SOL":
            return CryptoTokenomics(
                token_id="solana",
                symbol="SOL",
                circulating_supply=400e6,
                maximum_supply=None,
                total_supply=500e6,
                inflation_rate=0.05,
                emission_schedule="decreasing inflation schedule",
                market_capitalization=60e9,
                fully_diluted_valuation=75e9,
                burn_mechanism=False,
                staking_enabled=True,
                staking_rate=0.07,
                governance_token=True,
                timestamp=now_utc(),
            )
        return None

    def get_token_unlocks(self, symbol: str) -> List[TokenUnlock]:
        sym = symbol.upper()
        if sym == "SOL":
            return [
                TokenUnlock(
                    unlock_id="unlock_1",
                    symbol="SOL",
                    unlock_date=now_utc() + timedelta(days=30),
                    amount=5e6,
                    percentage_of_supply=1.0,
                    source="team",
                    cliff_vesting=True,
                )
            ]
        return []

    def get_on_chain_metrics(self, symbol: str) -> Optional[OnChainMetrics]:
        sym = symbol.upper()
        if sym == "BTC":
            return OnChainMetrics(
                token_id="bitcoin",
                symbol="BTC",
                active_addresses_24h=900000,
                transaction_count_24h=300000,
                transaction_volume_24h=5e9,
                average_transaction_value=16666.0,
                exchange_inflow_24h=5000,
                exchange_outflow_24h=4500,
                net_exchange_flow=-500,
                whale_transactions_24h=100,
                large_holders_count=10000,
                hashrate=500e18,  # EH/s
                staked_amount=None,
                tvl=None,
                protocol_revenue_24h=None,
                timestamp=now_utc(),
            )
        elif sym == "ETH":
            return OnChainMetrics(
                token_id="ethereum",
                symbol="ETH",
                active_addresses_24h=500000,
                transaction_count_24h=1200000,
                transaction_volume_24h=3e9,
                average_transaction_value=2500.0,
                exchange_inflow_24h=10000,
                exchange_outflow_24h=11000,
                net_exchange_flow=1000,
                whale_transactions_24h=150,
                large_holders_count=5000,
                hashrate=None,
                staked_amount=30e6,
                tvl=30e9,
                protocol_revenue_24h=10e6,
                timestamp=now_utc(),
            )
        return None
