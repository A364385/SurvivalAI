from typing import List, Optional
from app.core.models.agent import AgentResult
from app.core.models.deep_research import (
    CompletenessLevel, Contradiction, ContradictionStatus, FinancialHealthMetrics,
)
from app.core.models.market import MarketSnapshot, MarketTrend


class DossierContradictionDetector:
    """Surface opposing evidence. Never drop a contradiction to make the thesis look cleaner."""

    def detect(
        self,
        health: FinancialHealthMetrics,
        snapshot: Optional[MarketSnapshot],
        news_result: Optional[AgentResult],
        crisis_result: Optional[AgentResult],
    ) -> List[Contradiction]:
        found: List[Contradiction] = []
        if (
            health.revenue_growth is not None
            and health.free_cash_flow_growth is not None
            and health.revenue_growth > 0
            and health.free_cash_flow_growth < 0
        ):
            found.append(Contradiction(
                claim_a="Revenue growth is positive.",
                claim_b="Free cash flow growth is negative.",
                evidence_a=f"revenue_growth={health.revenue_growth}",
                evidence_b=f"free_cash_flow_growth={health.free_cash_flow_growth}",
                severity="HIGH",
                status=ContradictionStatus.UNRESOLVED,
                confidence=0.9,
            ))
        if snapshot is not None and crisis_result is not None:
            events = (crisis_result.analysis or {}).get("events") or []
            high_geo = any(
                str(e.get("risk", {}).get("dimensions", {}).get("overall_risk_level", "")).upper()
                in ("HIGH", "SEVERE", "CRITICAL")
                for e in events if isinstance(e, dict)
            )
            if snapshot.trend == MarketTrend.UPTREND and high_geo:
                found.append(Contradiction(
                    claim_a="Market trend is classified as UPTREND.",
                    claim_b="Geopolitical/crisis overall risk is elevated.",
                    evidence_a=f"trend={snapshot.trend.value}",
                    evidence_b="crisis overall_risk_level is HIGH or above",
                    severity="MODERATE",
                    status=ContradictionStatus.UNRESOLVED,
                    confidence=0.75,
                ))
        if news_result is not None and crisis_result is not None:
            news_dir = str((news_result.impact or {}).get("direction", "")).upper()
            crisis_dir = str((crisis_result.impact or {}).get("direction", "")).upper()
            if news_dir == "POSITIVE" and crisis_dir == "NEGATIVE":
                found.append(Contradiction(
                    claim_a="News research impact direction is POSITIVE.",
                    claim_b="Crisis research impact direction is NEGATIVE.",
                    evidence_a="news_research.impact.direction",
                    evidence_b="crisis_research.impact.direction",
                    severity="MODERATE",
                    status=ContradictionStatus.UNRESOLVED,
                    confidence=0.7,
                ))
        if news_result is not None and getattr(news_result, "warnings", None):
            for w in news_result.warnings:
                if "conflict" in w.lower() or "discrepancy" in w.lower():
                    found.append(Contradiction(
                        claim_a="News cluster contains conflicting reporting.",
                        claim_b=w,
                        evidence_a="news_research.warnings",
                        evidence_b=w,
                        severity="MODERATE",
                        status=ContradictionStatus.UNRESOLVED,
                        confidence=0.65,
                    ))
                    break
        return found


def completeness_from_presence(present: bool, stale: bool = False, conflicting: bool = False, partial: bool = False) -> CompletenessLevel:
    if conflicting:
        return CompletenessLevel.CONFLICTING
    if stale:
        return CompletenessLevel.STALE
    if not present:
        return CompletenessLevel.MISSING
    if partial:
        return CompletenessLevel.PARTIAL
    return CompletenessLevel.COMPLETE
