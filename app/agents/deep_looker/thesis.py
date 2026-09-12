from typing import List, Optional
from app.core.models.deep_research import (
    CompletenessLevel, Contradiction, ContradictionStatus, DataCompleteness,
    FinancialHealthMetrics, IdentifiedRisk, InvestmentThesis, ProbabilityAssessment,
    RiskCategory, ScenarioCase, ThesisAssumption, ThesisStatus,
)
from app.core.models.market import AssetCapabilities, MarketSnapshot, MarketTrend


class ThesisBuilder:
    """Builds an investment thesis narrative without issuing BUY/SELL/HOLD."""

    def build(
        self,
        symbol: str,
        capabilities: AssetCapabilities,
        health: FinancialHealthMetrics,
        snapshot: Optional[MarketSnapshot],
        contradictions: List[Contradiction],
        risks: List[IdentifiedRisk],
        completeness: DataCompleteness,
        supporting: List[str],
        invalidating: List[str],
    ) -> InvestmentThesis:
        unresolved = [c for c in contradictions if c.status == ContradictionStatus.UNRESOLVED]
        data_thin = completeness.fundamentals in (
            CompletenessLevel.MISSING, CompletenessLevel.INSUFFICIENT,
        ) and completeness.market_data in (
            CompletenessLevel.MISSING, CompletenessLevel.INSUFFICIENT,
        )
        if data_thin and not supporting:
            status = ThesisStatus.INSUFFICIENT_DATA
        elif unresolved and any(c.severity in ("HIGH", "SEVERE", "CRITICAL") for c in unresolved):
            status = ThesisStatus.MIXED
        elif len(invalidating) > len(supporting) + 1:
            status = ThesisStatus.WEAK_SUPPORT
        elif completeness.fundamentals == CompletenessLevel.COMPLETE and snapshot and snapshot.trend == MarketTrend.UPTREND and not unresolved and supporting:
            status = ThesisStatus.STRONG_SUPPORT
        elif supporting and not invalidating:
            status = ThesisStatus.MODERATE_SUPPORT
        elif supporting or snapshot is not None:
            status = ThesisStatus.MIXED if unresolved else ThesisStatus.MODERATE_SUPPORT
        else:
            status = ThesisStatus.INSUFFICIENT_DATA

        story = (
            f"{symbol} ({capabilities.asset_type}) is under due-diligence review. "
            "This thesis is research context for later decision agents, not an approval to invest."
        )
        assumptions = [
            ThesisAssumption(
                assumption="Available statements and market snapshots are correctly attributed to this symbol.",
                evidence="Provider identifiers and task symbol match.",
                confidence=0.7,
                invalidation_condition="Identifier mismatch or restated financials that reverse key metrics.",
            )
        ]
        if capabilities.has_financial_statements:
            assumptions.append(ThesisAssumption(
                assumption="Reported financial relationships (growth, leverage) persist until new filings.",
                evidence="Latest FinancialStatements period_end when present.",
                confidence=0.55 if health.revenue_growth is not None else 0.2,
                invalidation_condition="Revenue growth materially reverses in a subsequent reporting period.",
            ))
        invalidation = list(invalidating) or [
            "Material adverse filing restatement.",
            "Major regulatory restriction affecting the core activity.",
        ]
        if capabilities.has_financial_statements:
            invalidation.append("Debt increases beyond levels implied by current debt-to-equity/cash metrics when those metrics exist.")
        scenarios = [
            ScenarioCase(
                name="BULL",
                assumptions=["Supportive factors persist and unresolved contradictions are later explained by filings."],
                supporting_evidence=list(supporting) or ["Insufficient data for a detailed bull case."],
                risks=["Optimism is not evidence."],
                time_horizon="MEDIUM_TERM",
                confidence=0.35,
            ),
            ScenarioCase(
                name="BASE",
                assumptions=["Current data quality and contradictions remain as observed."],
                supporting_evidence=["Dossier completeness flags and deterministic metrics."],
                risks=[r.description for r in risks[:3]] or ["Unknown risks due to missing data."],
                time_horizon="MEDIUM_TERM",
                confidence=0.45,
            ),
            ScenarioCase(
                name="BEAR",
                assumptions=["Invalidation conditions occur."],
                supporting_evidence=list(invalidating) or ["No specific bear evidence beyond missing data."],
                risks=["Thesis does not survive a material adverse change."],
                time_horizon="SHORT_TERM",
                confidence=0.4,
            ),
        ]
        return InvestmentThesis(
            story=story,
            supporting_factors=supporting,
            invalidation_factors=invalidating,
            what_would_strengthen=["Independent confirmation of currently one-source claims.", "Complete financials with peer valuation context."],
            what_would_weaken=invalidation,
            status=status,
            assumptions=assumptions,
            scenarios=scenarios,
            invalidation_conditions=list(dict.fromkeys(invalidation)),
        )


def collect_risks(
    capabilities: AssetCapabilities,
    health: FinancialHealthMetrics,
    snapshot: Optional[MarketSnapshot],
    crisis_events: list,
    news_impact_direction: Optional[str],
) -> List[IdentifiedRisk]:
    risks: List[IdentifiedRisk] = []
    if snapshot is not None and snapshot.volatility.short_term_volatility is not None:
        vol = snapshot.volatility.short_term_volatility
        if vol >= 0.4:
            risks.append(IdentifiedRisk(
                category=RiskCategory.MARKET,
                description="Short-term historical volatility is elevated.",
                severity="HIGH",
                probability=ProbabilityAssessment.UNKNOWN,
                potential_impact="Wider drawdowns in quoted prices.",
                time_horizon="SHORT_TERM",
                evidence_ids=["market_snapshot"],
                confidence=0.8,
            ))
        if snapshot.volume.is_unusual_volume:
            risks.append(IdentifiedRisk(
                category=RiskCategory.LIQUIDITY,
                description="Volume is unusual versus the recent average.",
                severity="MODERATE",
                probability=ProbabilityAssessment.UNKNOWN,
                potential_impact="Less reliable liquidity at the last print.",
                time_horizon="IMMEDIATE",
                evidence_ids=["market_snapshot"],
                confidence=0.7,
            ))
    if health.debt_to_equity is not None and health.debt_to_equity >= 2.0:
        risks.append(IdentifiedRisk(
            category=RiskCategory.FINANCIAL,
            description="Debt-to-equity is elevated versus a simple 2.0 observational threshold.",
            severity="HIGH",
            probability=ProbabilityAssessment.UNKNOWN,
            potential_impact="Greater sensitivity to refinancing and earnings shocks.",
            time_horizon="MEDIUM_TERM",
            evidence_ids=["financial_statements"],
            confidence=0.75,
        ))
    if not capabilities.has_financial_statements:
        risks.append(IdentifiedRisk(
            category=RiskCategory.BUSINESS,
            description="Issuer-style financial statements are not applicable or not available for this asset type.",
            severity="MODERATE",
            probability=ProbabilityAssessment.UNKNOWN,
            potential_impact="Fundamental health cannot be verified from filings.",
            time_horizon="UNKNOWN",
            evidence_ids=["asset_capabilities"],
            confidence=0.85,
        ))
    for ev in crisis_events:
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("event_type", "OTHER"))
        if et in ("SANCTIONS", "REGULATION"):
            risks.append(IdentifiedRisk(
                category=RiskCategory.REGULATORY if et == "REGULATION" else RiskCategory.GEOPOLITICAL,
                description=f"Crisis research reports {et}.",
                severity=str(ev.get("severity", "UNKNOWN")),
                probability=ProbabilityAssessment.UNKNOWN,
                potential_impact="Policy or access constraints.",
                time_horizon=str(ev.get("horizon", "UNKNOWN")),
                evidence_ids=[str(ev.get("event_id", "crisis"))],
                confidence=float(ev.get("confidence", 0.5) or 0.5),
            ))
        if et in ("SUPPLY_CHAIN_DISRUPTION", "ENERGY_DISRUPTION"):
            risks.append(IdentifiedRisk(
                category=RiskCategory.SUPPLY_CHAIN if et == "SUPPLY_CHAIN_DISRUPTION" else RiskCategory.GEOPOLITICAL,
                description=f"Crisis research reports {et}.",
                severity=str(ev.get("severity", "UNKNOWN")),
                probability=ProbabilityAssessment.UNKNOWN,
                potential_impact="Input, energy, or logistics disruption.",
                time_horizon=str(ev.get("horizon", "UNKNOWN")),
                evidence_ids=[str(ev.get("event_id", "crisis"))],
                confidence=float(ev.get("confidence", 0.5) or 0.5),
            ))
    return risks
