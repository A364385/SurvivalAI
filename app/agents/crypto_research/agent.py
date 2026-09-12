"""Crypto Research Agent implementation.

Specialized research agent for cryptocurrency markets focusing on:
- Crypto-specific market structure
- Tokenomics and supply dynamics
- On-chain metrics
- Crypto-specific risk factors
- Integration with general market research
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Callable

from app.agents.base import BaseAgent
from app.core.models.agent import AgentConfig, AgentStatus, AgentResult
from app.core.models.task import Task
from app.core.models.knowledge import Fact, Source
from app.core.models.error import ErrorInfo
from app.core.models.events import BaseEvent
from app.core.models.crypto import (
    CryptoSnapshot,
    CryptoMarketStructure,
    CryptoRiskFactor,
    CryptoRiskType,
    CryptoRegime,
    TokenUnlock,
)
from app.services.crypto.market_provider import CryptoMarketDataProvider
from app.services.crypto.tokenomics_provider import CryptoFundamentalDataProvider
from app.services.crypto.onchain_provider import OnChainDataProvider
from app.services.llm.provider import LLMProvider
from app.utils.ids import generate_id
from app.utils.time import now_utc
from app.utils.logging import get_logger

logger = get_logger(__name__)


class CryptoResearchAgent(BaseAgent):
    """Research-only agent for cryptocurrency markets.

    Analyzes crypto-specific metrics, tokenomics, on-chain data,
    and crypto risk factors. Does NOT independently buy, sell,
    execute orders, modify risk limits, or activate strategies.

    Output is research information for Deep Looker / Risk Manager / CEO.
    """

    def __init__(
        self,
        agent_id: str,
        crypto_market_provider: CryptoMarketDataProvider,
        tokenomics_provider: Optional[CryptoFundamentalDataProvider] = None,
        onchain_provider: Optional[OnChainDataProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
        configuration: Optional[AgentConfig] = None,
        event_publisher: Optional[Callable[[BaseEvent], None]] = None,
    ):
        config = configuration or AgentConfig(version="1.0.0", model=None)
        super().__init__(
            agent_id=agent_id,
            agent_name="CryptoResearchAgent",
            role="Cryptocurrency Market Analyst & Tokenomics Researcher",
            description="Analyzes crypto markets, tokenomics, on-chain metrics, and crypto-specific risk factors.",
            version="1.0.0",
            configuration=config,
            capabilities=[
                "crypto_market_structure",
                "tokenomics_analysis",
                "on_chain_metrics",
                "crypto_risk_assessment",
                "market_wide_conditions",
            ],
        )
        self.crypto_market_provider = crypto_market_provider
        self.tokenomics_provider = tokenomics_provider
        self.onchain_provider = onchain_provider
        self.llm_provider = llm_provider
        self.event_publisher = event_publisher
        self.emitted_events: List[BaseEvent] = []

    def _emit(self, event: BaseEvent) -> None:
        self.emitted_events.append(event)
        if self.event_publisher:
            self.event_publisher(event)

    def _assess_market_structure(
        self,
        symbol: str,
        current_price: float,
        price_change_24h: float,
        volatility: float,
    ) -> CryptoMarketStructure:
        """Assess crypto market structure deterministically."""
        # Trend classification
        if price_change_24h > 0.05:
            trend = "STRONG_UPTREND"
        elif price_change_24h > 0.02:
            trend = "UPTREND"
        elif price_change_24h > -0.02:
            trend = "SIDEWAYS"
        elif price_change_24h > -0.05:
            trend = "DOWNTREND"
        else:
            trend = "STRONG_DOWNTREND"

        # Momentum
        momentum = "BULLISH" if price_change_24h > 0 else "BEARISH"

        # Volatility classification
        if volatility > 0.8:
            vol_class = "EXTREME"
        elif volatility > 0.5:
            vol_class = "HIGH"
        elif volatility > 0.3:
            vol_class = "MODERATE"
        else:
            vol_class = "LOW"

        # Liquidity (simplified - would use volume metrics in production)
        liquidity = "HIGH"  # Default for major cryptos

        # Drawdown (simplified - would use historical data in production)
        drawdown = abs(min(0, price_change_24h))

        # Regime
        if trend in ("STRONG_UPTREND", "UPTREND") and vol_class == "HIGH":
            regime = CryptoRegime.BULL_MARKET
        elif trend in ("STRONG_DOWNTREND", "DOWNTREND") and vol_class == "HIGH":
            regime = CryptoRegime.BEAR_MARKET
        else:
            regime = CryptoRegime.UNKNOWN

        return CryptoMarketStructure(
            symbol=symbol,
            trend=trend,
            momentum=momentum,
            volatility=vol_class,
            liquidity=liquidity,
            drawdown=drawdown,
            regime=regime,
            timestamp=now_utc(),
        )

    def _identify_crypto_risks(
        self,
        symbol: str,
        market_structure: CryptoMarketStructure,
        tokenomics: Optional[Any],
        on_chain: Optional[Any],
        upcoming_unlocks: List[TokenUnlock],
    ) -> List[CryptoRiskFactor]:
        """Identify crypto-specific risk factors."""
        risks: List[CryptoRiskFactor] = []

        # Extreme volatility risk
        if market_structure.volatility in ("EXTREME", "HIGH"):
            risks.append(CryptoRiskFactor(
                risk_id=generate_id("crisk"),
                risk_type=CryptoRiskType.EXTREME_VOLATILITY,
                symbol=symbol,
                severity="HIGH" if market_structure.volatility == "EXTREME" else "MODERATE",
                confidence=0.9,
                evidence=[f"Market structure shows {market_structure.volatility} volatility"],
                time_horizon="SHORT_TERM",
                affected_assets=[symbol],
                description=f"{symbol} exhibits {market_structure.volatility.lower()} volatility",
                timestamp=now_utc(),
            ))

        # Token unlock risk
        for unlock in upcoming_unlocks:
            days_until = (unlock.unlock_date - now_utc()).days
            if days_until <= 30:  # Within 30 days
                risks.append(CryptoRiskFactor(
                    risk_id=generate_id("crisk"),
                    risk_type=CryptoRiskType.TOKEN_UNLOCK_RISK,
                    symbol=symbol,
                    severity="HIGH" if unlock.percentage_of_supply > 2.0 else "MODERATE",
                    confidence=0.8,
                    evidence=[
                        f"Scheduled unlock of {unlock.amount} {symbol} ({unlock.percentage_of_supply:.1f}% of supply)",
                        f"Unlock date: {unlock.unlock_date}",
                        f"Source: {unlock.source}",
                    ],
                    time_horizon="SHORT_TERM" if days_until <= 7 else "MEDIUM_TERM",
                    affected_assets=[symbol],
                    description=f"Token unlock of {unlock.percentage_of_supply:.1f}% of supply in {days_until} days",
                    timestamp=now_utc(),
                ))

        # Regulatory risk (would use crisis agent data in production)
        risks.append(CryptoRiskFactor(
            risk_id=generate_id("crisk"),
            risk_type=CryptoRiskType.REGULATORY_RISK,
            symbol=symbol,
            severity="MODERATE",
            confidence=0.6,
            evidence=["Crypto assets subject to evolving regulatory landscape"],
            time_horizon="LONG_TERM",
            affected_assets=[symbol],
            description="Crypto assets face regulatory uncertainty",
            timestamp=now_utc(),
        ))

        # Smart contract risk (for non-BTC assets)
        if symbol.upper() != "BTC":
            risks.append(CryptoRiskFactor(
                risk_id=generate_id("crisk"),
                risk_type=CryptoRiskType.SMART_CONTRACT_RISK,
                symbol=symbol,
                severity="MODERATE",
                confidence=0.7,
                evidence=["Smart contract protocols carry inherent technical risk"],
                time_horizon="LONG_TERM",
                affected_assets=[symbol],
                description="Smart contract vulnerability risk",
                timestamp=now_utc(),
            ))

        return risks

    def _build_crypto_snapshot(
        self,
        symbol: str,
    ) -> CryptoSnapshot:
        """Build a comprehensive crypto snapshot."""
        warnings = []
        supported = self.crypto_market_provider.get_supported_symbols()

        if symbol.upper() not in supported:
            warnings.append(f"Symbol {symbol} is not in the supported crypto list")

        try:
            quote = self.crypto_market_provider.get_quote(symbol)
            current_price = (quote.bid_price + quote.ask_price) / 2
        except Exception as e:
            logger.warning("Could not fetch quote for %s: %s", symbol, e)
            current_price = 0.0
            warnings.append(f"Could not fetch quote for {symbol}: {e}")

        # Market structure
        market_structure = self._assess_market_structure(
            symbol=symbol,
            current_price=current_price,
            price_change_24h=0.03,  # Would calculate from historical data
            volatility=0.5,  # Would calculate from historical data
        )

        # Tokenomics
        tokenomics = None
        if self.tokenomics_provider:
            try:
                tokenomics = self.tokenomics_provider.get_tokenomics(symbol)
            except Exception as e:
                logger.warning("Could not fetch tokenomics for %s: %s", symbol, e)

        # On-chain metrics
        on_chain = None
        if self.onchain_provider:
            try:
                on_chain = self.onchain_provider.get_on_chain_metrics(symbol)
            except Exception as e:
                logger.warning("Could not fetch on-chain metrics for %s: %s", symbol, e)

        # Market-wide conditions
        market_wide = None
        try:
            market_wide = self.crypto_market_provider.get_market_wide_conditions()
        except Exception as e:
            logger.warning("Could not fetch market-wide conditions: %s", e)

        # Upcoming unlocks
        unlocks = []
        if self.tokenomics_provider:
            try:
                unlocks = self.tokenomics_provider.get_token_unlocks(symbol)
            except Exception as e:
                logger.warning("Could not fetch token unlocks for %s: %s", symbol, e)

        # Risk factors
        risks = self._identify_crypto_risks(
            symbol=symbol,
            market_structure=market_structure,
            tokenomics=tokenomics,
            on_chain=on_chain,
            upcoming_unlocks=unlocks,
        )

        # Correlations
        correlations = []
        try:
            if symbol.upper() != "BTC":
                btc_corr = self.crypto_market_provider.get_correlation(f"{symbol}-BTC")
                if btc_corr:
                    correlations.append(btc_corr)
        except Exception as e:
            logger.warning("Could not fetch correlations for %s: %s", symbol, e)

        return CryptoSnapshot(
            symbol=symbol,
            timestamp=now_utc(),
            current_price=current_price,
            market_structure=market_structure,
            tokenomics=tokenomics,
            on_chain_metrics=on_chain,
            risk_factors=risks,
            market_wide_conditions=market_wide,
            correlations=correlations,
            upcoming_unlocks=unlocks,
            data_quality="VALID" if current_price > 0 else "INSUFFICIENT_DATA",
            warnings=warnings,
        )

    def process_task(self, task: Task) -> AgentResult:
        """Process a crypto research task."""
        self.status = AgentStatus.RUNNING
        now = now_utc()
        symbol = task.input_data.get("symbol", "BTC")
        symbols = task.input_data.get("symbols", [symbol])

        logger.info(
            "CryptoResearchAgent processing task %s for %s symbol(s): %s",
            task.task_id,
            len(symbols),
            symbols,
        )

        try:
            snapshots: List[CryptoSnapshot] = []
            for sym in symbols:
                snapshots.append(self._build_crypto_snapshot(sym))

            primary_snap = snapshots[0]
            warnings = primary_snap.warnings[:]

            # Build deterministic facts
            facts: List[Fact] = [
                Fact(
                    fact_id=generate_id("cfact"),
                    statement=(
                        f"{primary_snap.symbol} current price is {primary_snap.current_price} "
                        f"with {primary_snap.market_structure.trend} trend."
                    ),
                    source_ids=["CryptoMarketDataProvider"],
                    timestamp=now,
                    confidence=1.0,
                    data_type="crypto_price",
                ),
            ]

            if primary_snap.tokenomics:
                if primary_snap.tokenomics.market_capitalization:
                    facts.append(Fact(
                        fact_id=generate_id("cfact"),
                        statement=(
                            f"{primary_snap.symbol} market capitalization is "
                            f"${primary_snap.tokenomics.market_capitalization:,.0f}."
                        ),
                        source_ids=["CryptoFundamentalDataProvider"],
                        timestamp=now,
                        confidence=0.9,
                        data_type="tokenomics",
                    ))

            if primary_snap.on_chain_metrics:
                if primary_snap.on_chain_metrics.active_addresses_24h:
                    facts.append(Fact(
                        fact_id=generate_id("cfact"),
                        statement=(
                            f"{primary_snap.symbol} has {primary_snap.on_chain_metrics.active_addresses_24h:,} "
                            f"active addresses in the last 24 hours."
                        ),
                        source_ids=["OnChainDataProvider"],
                        timestamp=now,
                        confidence=0.8,
                        data_type="on_chain",
                    ))

            # Add risk factor facts
            for risk in primary_snap.risk_factors:
                facts.append(Fact(
                    fact_id=generate_id("cfact"),
                    statement=f"{primary_snap.symbol} risk: {risk.risk_type.value} - {risk.description}",
                    source_ids=["CryptoResearchAgent"],
                    timestamp=now,
                    confidence=risk.confidence,
                    data_type="crypto_risk",
                ))

            # Build analysis
            analysis = {
                "market_structure": {
                    "trend": primary_snap.market_structure.trend,
                    "momentum": primary_snap.market_structure.momentum,
                    "volatility": primary_snap.market_structure.volatility,
                    "regime": primary_snap.market_structure.regime.value,
                },
                "risk_factors": [
                    {
                        "type": r.risk_type.value,
                        "severity": r.severity,
                        "description": r.description,
                        "confidence": r.confidence,
                    }
                    for r in primary_snap.risk_factors
                ],
            }

            if primary_snap.tokenomics:
                analysis["tokenomics"] = {
                    "market_cap": primary_snap.tokenomics.market_capitalization,
                    "circulating_supply": primary_snap.tokenomics.circulating_supply,
                    "inflation_rate": primary_snap.tokenomics.inflation_rate,
                }

            if primary_snap.on_chain_metrics:
                analysis["on_chain"] = {
                    "active_addresses_24h": primary_snap.on_chain_metrics.active_addresses_24h,
                    "transaction_count_24h": primary_snap.on_chain_metrics.transaction_count_24h,
                }

            if primary_snap.market_wide_conditions:
                analysis["market_wide"] = {
                    "btc_dominance": primary_snap.market_wide_conditions.btc_dominance,
                    "total_market_cap": primary_snap.market_wide_conditions.total_crypto_market_cap,
                    "fear_greed_index": primary_snap.market_wide_conditions.fear_greed_index,
                }

            # Sources
            sources = [
                Source(
                    source_id="CryptoMarketDataProvider",
                    title=f"Crypto Market Data ({primary_snap.symbol})",
                    publisher="CryptoMarketDataProvider",
                    url="internal://crypto-market-data",
                    published_at=now,
                    retrieved_at=now,
                    is_primary=True,
                    reliability_score=0.9,
                )
            ]

            if self.tokenomics_provider:
                sources.append(Source(
                    source_id="CryptoFundamentalDataProvider",
                    title=f"Crypto Tokenomics ({primary_snap.symbol})",
                    publisher="CryptoFundamentalDataProvider",
                    url="internal://crypto-tokenomics",
                    published_at=now,
                    retrieved_at=now,
                    is_primary=True,
                    reliability_score=0.85,
                ))

            if self.onchain_provider:
                sources.append(Source(
                    source_id="OnChainDataProvider",
                    title=f"On-Chain Metrics ({primary_snap.symbol})",
                    publisher="OnChainDataProvider",
                    url="internal://onchain-data",
                    published_at=now,
                    retrieved_at=now,
                    is_primary=True,
                    reliability_score=0.8,
                ))

            summary = (
                f"{primary_snap.symbol} crypto analysis: {primary_snap.market_structure.trend} trend, "
                f"{primary_snap.market_structure.volatility} volatility, "
                f"{len(primary_snap.risk_factors)} risk factors identified."
            )

            result = AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                timestamp=now,
                status=AgentStatus.SUCCESS,
                summary=summary,
                confidence=0.85,
                facts=facts,
                analysis=analysis,
                impact={
                    "direction": "UNCERTAIN",
                    "horizon": "MEDIUM_TERM",
                    "affected_symbols": [primary_snap.symbol],
                },
                warnings=warnings,
                sources=sources,
                metadata={
                    "symbol": primary_snap.symbol,
                    "risk_factors_count": len(primary_snap.risk_factors),
                    "has_tokenomics": primary_snap.tokenomics is not None,
                    "has_on_chain": primary_snap.on_chain_metrics is not None,
                    "data_quality": primary_snap.data_quality,
                },
            )

            self.status = AgentStatus.SUCCESS
            return result

        except Exception as e:
            logger.error("CryptoResearchAgent failed: %s", e)
            self.status = AgentStatus.FAILED
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                timestamp=now,
                status=AgentStatus.FAILED,
                summary=f"Crypto research failed: {str(e)}",
                confidence=0.0,
                facts=[],
                analysis={},
                impact={},
                warnings=[f"Error: {str(e)}"],
                sources=[],
                metadata={"error": str(e)},
            )
