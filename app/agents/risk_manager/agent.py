from dataclasses import asdict
from datetime import timedelta
from math import isfinite
from typing import Any, Callable, Dict, List, Optional

from app.agents.base import BaseAgent
from app.agents.market_research.recommendation_guard import sanitize_llm_payload
from app.core.memory.store import MemoryStore
from app.core.models.agent import AgentConfig, AgentResult, AgentStatus
from app.core.models.deep_research import DeepResearchDossier
from app.core.models.error import ErrorInfo
from app.core.models.events import (
    BaseEvent,
    RiskAssessmentCompleted,
    RiskAssessmentStarted,
    RiskViolationEvent,
)
from app.core.models.execution import PositionState
from app.core.models.knowledge import Evidence
from app.core.models.market import MarketSnapshot
from app.core.models.memory import MemoryRecord, MemoryType
from app.core.models.risk import (
    InvestmentProposal,
    PortfolioRiskState,
    PositionExposure,
    RiskAssessment,
    RiskDecision,
    RiskDimension,
    RiskPolicy,
    RiskRuleResult,
    RiskRuleType,
    RiskSeverity,
)
from app.core.models.task import Task
from app.services.llm.provider import LLMProvider
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


_SEVERITY_RANK = {
    RiskSeverity.UNKNOWN: 0,
    RiskSeverity.LOW: 1,
    RiskSeverity.MODERATE: 2,
    RiskSeverity.HIGH: 3,
    RiskSeverity.CRITICAL: 4,
}


class RiskManagerAgent(BaseAgent):
    """Deterministic-first risk gate.

    The Risk Manager evaluates whether a proposal fits configured risk limits. It is not a
    strategist and has no execution or portfolio mutation authority.
    """

    SYSTEM_PROMPT = """You are the SurvivalAI Chief Risk Officer (CRO). Your role is to evaluate investment proposals against the organization's Risk Policy.

Your Goal:
1. Analyze the Investment Proposal (asset, value, quantity).
2. Evaluate it against the Portfolio State (current cash, existing positions, concentration).
3. Check against the provided Risk Policy (limits on position size, concentration, cash reserve, and sector exposure).
4. Incorporate external context (Market Snapshots, Deep Research, News, Crisis data).
5. Make a final Decision:
   - APPROVED: The proposal is safe and aligns with policy.
   - BLOCKED: The proposal violates a HARD risk limit (e.g., too much concentration, too little cash reserve).
   - INSUFFICIENT_DATA: The proposal lacks critical information to make a safe decision.

Safety Rules:
- If a HARD limit in the policy is violated, you MUST return 'BLOCKED'.
- You must provide a clear, logical reasoning for every decision.
- Identify specific Risk Dimensions (Position Size, Concentration, Sector Exposure, Volatility, Liquidity, Crisis, Fundamental, Correlation).
- Be conservative. If the context is ambiguous, lean towards 'INSUFFICIENT_DATA' or 'APPROVED_WITH_WARNINGS'.

Output Format:
You must output a valid JSON object with the following keys:
{
  "decision": "APPROVED" | "BLOCKED" | "INSUFFICIENT_DATA",
  "reasoning": "A detailed explanation of your logic...",
  "confidence": 0.0 to 1.0,
  "risk_dimensions": {
    "position_size": {"status": "PASS" | "FAIL" | "WARNING", "value": float, "message": "reason"},
    "concentration": {"status": "PASS" | "FAIL" | "WARNING", "value": float, "message": "reason"},
    "cash_reserve": {"status": "PASS" | "FAIL" | "WARNING", "value": float, "message": "reason"},
    "sector_exposure": {"status": "PASS" | "FAIL" | "WARNING", "value": float, "message": "reason"},
    "volatility": {"status": "PASS" | "FAIL" | "WARNING", "value": float, "message": "reason"},
    "liquidity": {"status": "PASS" | "FAIL" | "WARNING", "value": float, "message": "reason"},
    "crisis": {"status": "PASS" | "FAIL" | "WARNING", "value": float, "message": "reason"},
    "fundamental": {"status": "PASS" | "FAIL" | "WARNING", "value": float, "message": "reason"},
    "correlation": {"status": "PASS" | "FAIL" | "WARNING", "value": float, "message": "reason"}
  },
  "warnings": ["list of warnings"],
  "required_actions": ["list of actions for the user"]
}
"""

    def __init__(
        self,
        agent_id: str,
        llm_provider: Optional[LLMProvider] = None,
        memory_store: Optional[MemoryStore] = None,
        configuration: Optional[AgentConfig] = None,
        policy: Optional[RiskPolicy] = None,
        event_publisher: Optional[Callable[[BaseEvent], None]] = None,
    ):
        config = configuration or AgentConfig(version="1.0.0", model=None, max_tokens=1200)
        super().__init__(
            agent_id=agent_id,
            agent_name="RiskManagerAgent",
            role="Deterministic Portfolio Risk Gate",
            description="Approves, warns, blocks, or marks investment proposals as insufficient data based on configured risk limits.",
            version="1.0.0",
            configuration=config,
            capabilities=[
                "position_size_risk",
                "cash_reserve_protection",
                "concentration_limits",
                "volatility_risk",
                "crisis_risk",
                "deep_research_risk",
                "correlation_risk",
                "risk_event_emission",
            ],
        )
        self.llm_provider = llm_provider
        self.memory_store = memory_store
        self.policy = policy or RiskPolicy()
        self.event_publisher = event_publisher
        self.emitted_events: List[BaseEvent] = []

    def _emit(self, event: BaseEvent) -> None:
        self.emitted_events.append(event)
        if self.event_publisher:
            self.event_publisher(event)

    def _proposal_from_input(self, data: Dict[str, Any]) -> InvestmentProposal:
        raw = data.get("proposal") or data.get("investment_proposal") or data
        if isinstance(raw, InvestmentProposal):
            return raw
        asset = str(raw.get("asset") or raw.get("symbol") or "").strip().upper()
        if not asset:
            raise ValueError("Risk assessment requires proposal.asset or proposal.symbol.")
        proposed_value = raw.get("proposed_position_value", raw.get("position_value", raw.get("amount")))
        return InvestmentProposal(
            proposal_id=str(raw.get("proposal_id") or raw.get("investment_id") or generate_id("proposal")),
            investment_id=raw.get("investment_id"),
            asset=asset,
            proposed_position_value=float(proposed_value),
            asset_class=raw.get("asset_class"),
            sector=raw.get("sector"),
            geography=raw.get("geography") or raw.get("country"),
            quantity=raw.get("quantity"),
            limit_price=raw.get("limit_price"),
            metadata=dict(raw.get("metadata", {})),
        )

    def _policy_from_input(self, data: Dict[str, Any]) -> RiskPolicy:
        raw = data.get("risk_policy") or data.get("policy")
        if raw is None:
            return self.policy
        if isinstance(raw, RiskPolicy):
            return raw
        merged = asdict(self.policy)
        merged.update(raw)
        return RiskPolicy(**merged)

    def _portfolio_from_input(self, data: Dict[str, Any]) -> PortfolioRiskState:
        raw = data.get("portfolio_state") or data.get("portfolio")
        if isinstance(raw, PortfolioRiskState):
            return raw
        if raw is None:
            raise ValueError("Risk assessment requires portfolio_state.")
        positions_raw = raw.get("positions", []) if isinstance(raw, dict) else getattr(raw, "positions", [])
        if isinstance(positions_raw, dict):
            positions_raw = list(positions_raw.values())
        positions = [self._position_exposure(p) for p in positions_raw]
        return PortfolioRiskState(
            portfolio_value=float(raw.get("portfolio_value", raw.get("equity")) if isinstance(raw, dict) else getattr(raw, "equity")),
            available_cash=float(raw.get("available_cash", raw.get("cash")) if isinstance(raw, dict) else getattr(raw, "cash")),
            positions=positions,
            historical_peak_value=raw.get("historical_peak_value") if isinstance(raw, dict) else getattr(raw, "historical_peak_value", None),
            timestamp=raw.get("timestamp") if isinstance(raw, dict) else getattr(raw, "last_synced_at", None),
            metadata=dict(raw.get("metadata", {})) if isinstance(raw, dict) else {},
        )

    def _position_exposure(self, raw: Any) -> PositionExposure:
        if isinstance(raw, PositionExposure):
            return raw
        if isinstance(raw, PositionState):
            return PositionExposure(symbol=raw.symbol, market_value=float(raw.market_value))
        return PositionExposure(
            symbol=str(raw.get("symbol", "")).upper(),
            market_value=float(raw.get("market_value", raw.get("value", 0.0))),
            asset_class=raw.get("asset_class"),
            sector=raw.get("sector"),
            geography=raw.get("geography") or raw.get("country"),
            correlation_group=raw.get("correlation_group"),
        )

    def _safe_pct(self, numerator: float, denominator: float) -> Optional[float]:
        if denominator <= 0:
            return None
        return numerator / denominator

    def _add_rule(
        self,
        passed: List[RiskRuleResult],
        violated: List[RiskRuleResult],
        rule: RiskRuleResult,
    ) -> None:
        if rule.passed:
            passed.append(rule)
        else:
            violated.append(rule)

    def _sum_exposure(self, positions: List[PositionExposure], attr: str, value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        total = 0.0
        has_classification = False
        for pos in positions:
            pos_value = getattr(pos, attr)
            if pos_value:
                has_classification = True
            if pos_value == value:
                total += pos.market_value
        if not has_classification:
            return None
        return total

    def _market_snapshot(self, data: Dict[str, Any]) -> Optional[MarketSnapshot]:
        snap = data.get("market_snapshot")
        if isinstance(snap, MarketSnapshot):
            return snap
        market_result = data.get("market_research_result")
        if isinstance(market_result, AgentResult):
            possible = market_result.analysis.get("market_snapshot")
            return possible if isinstance(possible, MarketSnapshot) else None
        return None

    def _deep_dossier(self, data: Dict[str, Any]) -> Optional[DeepResearchDossier]:
        dossier = data.get("deep_research_dossier")
        if isinstance(dossier, DeepResearchDossier):
            return dossier
        result = data.get("deep_looker_result") or data.get("deep_research_result")
        if isinstance(result, AgentResult):
            possible = result.analysis.get("dossier")
            return possible if isinstance(possible, DeepResearchDossier) else None
        return None

    def _severity_from_text(self, value: str) -> RiskSeverity:
        normalized = str(value or "UNKNOWN").upper()
        if normalized in ("SEVERE", "CRITICAL"):
            return RiskSeverity.CRITICAL
        if normalized == "HIGH":
            return RiskSeverity.HIGH
        if normalized == "MODERATE":
            return RiskSeverity.MODERATE
        if normalized == "LOW":
            return RiskSeverity.LOW
        return RiskSeverity.UNKNOWN

    def evaluate_investment(
        self,
        proposal: InvestmentProposal,
        portfolio: PortfolioRiskState,
        policy: Optional[RiskPolicy] = None,
        market_snapshot: Optional[MarketSnapshot] = None,
        news_result: Optional[AgentResult] = None,
        crisis_result: Optional[AgentResult] = None,
        deep_dossier: Optional[DeepResearchDossier] = None,
        generation_id: str = "gen_current",
    ) -> RiskAssessment:
        policy = policy or self.policy
        now = now_utc()
        passed: List[RiskRuleResult] = []
        violated: List[RiskRuleResult] = []
        warnings: List[str] = []
        required_actions: List[str] = []
        evidence = [
            Evidence("risk_policy", "Configured risk policy thresholds.", "RiskPolicy", 1.0),
            Evidence("portfolio_state", "Current portfolio value, cash, and positions.", "PortfolioRiskState", 1.0),
        ]

        if not isfinite(portfolio.portfolio_value) or portfolio.portfolio_value <= 0:
            raise ValueError("Portfolio value must be positive and finite.")
        if not isfinite(portfolio.available_cash) or portfolio.available_cash < 0:
            raise ValueError("Available cash must be non-negative and finite.")
        if not isfinite(proposal.proposed_position_value) or proposal.proposed_position_value <= 0:
            raise ValueError("Proposed position value must be positive and finite.")

        current_asset_value = sum(p.market_value for p in portfolio.positions if p.symbol == proposal.asset)
        resulting_asset_value = current_asset_value + proposal.proposed_position_value
        resulting_cash = portfolio.available_cash - proposal.proposed_position_value
        portfolio_exposure = self._safe_pct(proposal.proposed_position_value, portfolio.portfolio_value) or 0.0
        concentration_exposure = self._safe_pct(resulting_asset_value, portfolio.portfolio_value) or 0.0
        cash_reserve_pct = self._safe_pct(resulting_cash, portfolio.portfolio_value) or 0.0

        max_position_value = min(
            policy.max_single_position_pct * portfolio.portfolio_value,
            policy.max_portfolio_concentration_pct * portfolio.portfolio_value - current_asset_value,
            portfolio.available_cash - policy.min_cash_reserve_pct * portfolio.portfolio_value,
        )

        self._add_rule(passed, violated, RiskRuleResult(
            rule_id="max_single_position_pct",
            dimension=RiskDimension.POSITION_SIZE,
            rule_type=RiskRuleType.HARD,
            passed=portfolio_exposure <= policy.max_single_position_pct,
            description="Proposed position value must not exceed maximum single-position percentage.",
            observed_value=portfolio_exposure,
            limit_value=policy.max_single_position_pct,
            severity=RiskSeverity.HIGH,
            evidence_ids=["portfolio_state", "risk_policy"],
        ))
        self._add_rule(passed, violated, RiskRuleResult(
            rule_id="minimum_cash_reserve",
            dimension=RiskDimension.CASH_RESERVE,
            rule_type=RiskRuleType.HARD,
            passed=resulting_cash >= 0 and cash_reserve_pct >= policy.min_cash_reserve_pct,
            description="Resulting cash must preserve the configured minimum cash reserve.",
            observed_value=cash_reserve_pct,
            limit_value=policy.min_cash_reserve_pct,
            severity=RiskSeverity.CRITICAL,
            evidence_ids=["portfolio_state", "risk_policy"],
        ))
        self._add_rule(passed, violated, RiskRuleResult(
            rule_id="max_portfolio_concentration_pct",
            dimension=RiskDimension.CONCENTRATION,
            rule_type=RiskRuleType.HARD,
            passed=concentration_exposure <= policy.max_portfolio_concentration_pct,
            description="Resulting asset concentration must stay within configured maximum.",
            observed_value=concentration_exposure,
            limit_value=policy.max_portfolio_concentration_pct,
            severity=RiskSeverity.HIGH,
            evidence_ids=["portfolio_state", "risk_policy"],
        ))

        position_symbols = {p.symbol for p in portfolio.positions}
        resulting_position_count = len(position_symbols | {proposal.asset})
        self._add_rule(passed, violated, RiskRuleResult(
            rule_id="max_positions",
            dimension=RiskDimension.POSITION_COUNT,
            rule_type=RiskRuleType.HARD,
            passed=resulting_position_count <= policy.max_positions,
            description="Number of simultaneous positions must stay within policy.",
            observed_value=float(resulting_position_count),
            limit_value=float(policy.max_positions),
            severity=RiskSeverity.HIGH,
            evidence_ids=["portfolio_state", "risk_policy"],
        ))

        sector_exposure = self._classified_exposure(
            portfolio, proposal, "sector", policy.max_sector_exposure_pct,
            RiskDimension.SECTOR, "max_sector_exposure_pct", passed, violated, required_actions,
        )
        asset_class_exposure = self._classified_exposure(
            portfolio, proposal, "asset_class", policy.max_asset_class_exposure_pct,
            RiskDimension.ASSET_CLASS, "max_asset_class_exposure_pct", passed, violated, required_actions,
        )
        geographic_exposure = self._classified_exposure(
            portfolio, proposal, "geography", policy.max_geographic_exposure_pct,
            RiskDimension.GEOGRAPHIC, "max_geographic_exposure_pct", passed, violated, required_actions,
        )

        drawdown_risk = RiskSeverity.LOW
        if portfolio.historical_peak_value is not None and portfolio.historical_peak_value > 0:
            drawdown = (portfolio.historical_peak_value - portfolio.portfolio_value) / portfolio.historical_peak_value
            self._add_rule(passed, violated, RiskRuleResult(
                rule_id="max_portfolio_drawdown_pct",
                dimension=RiskDimension.DRAWDOWN,
                rule_type=RiskRuleType.HARD,
                passed=drawdown <= policy.max_portfolio_drawdown_pct,
                description="Portfolio drawdown must remain within configured safety limit.",
                observed_value=drawdown,
                limit_value=policy.max_portfolio_drawdown_pct,
                severity=RiskSeverity.CRITICAL,
                evidence_ids=["portfolio_state", "risk_policy"],
            ))
            drawdown_risk = RiskSeverity.HIGH if drawdown > policy.max_portfolio_drawdown_pct * 0.8 else RiskSeverity.LOW

        volatility_risk = self._volatility_check(market_snapshot, policy, passed, violated, warnings, required_actions, now)
        liquidity_risk = self._liquidity_check(market_snapshot, warnings)
        crisis_risk = self._crisis_check(crisis_result, policy, passed, violated, warnings)
        fundamental_risk, thesis_risk, data_quality_risk = self._deep_research_check(
            deep_dossier, policy, passed, violated, warnings, required_actions
        )
        correlation_risk = self._correlation_check(portfolio, proposal, policy, warnings)
        self._news_check(news_result, warnings)

        insufficient_rules = [r for r in violated if r.dimension == RiskDimension.DATA_QUALITY]
        hard_violations = [
            r for r in violated
            if r.rule_type == RiskRuleType.HARD and r.dimension != RiskDimension.DATA_QUALITY
        ]
        if hard_violations:
            decision = RiskDecision.BLOCKED
        elif insufficient_rules:
            decision = RiskDecision.INSUFFICIENT_DATA
        elif warnings or any(_SEVERITY_RANK[s] >= _SEVERITY_RANK[RiskSeverity.MODERATE] for s in (
            volatility_risk, liquidity_risk, crisis_risk, fundamental_risk, thesis_risk, correlation_risk, data_quality_risk
        )):
            decision = RiskDecision.APPROVED_WITH_WARNINGS
        else:
            decision = RiskDecision.APPROVED

        if hard_violations:
            required_actions.append("Do not send this proposal to execution unless a later reassessment passes hard rules.")
        if decision == RiskDecision.INSUFFICIENT_DATA:
            required_actions.append("Collect missing critical risk data before reassessment.")

        confidence = self._confidence(policy, market_snapshot, deep_dossier, crisis_result, violated)
        reasoning = self._reasoning(decision, violated, warnings)
        return RiskAssessment(
            assessment_id=generate_id("risk"),
            generation_id=generation_id,
            proposal_id=proposal.proposal_id,
            timestamp=now,
            asset=proposal.asset,
            proposed_position_size=proposal.quantity,
            proposed_position_value=proposal.proposed_position_value,
            portfolio_value=portfolio.portfolio_value,
            available_cash=portfolio.available_cash,
            resulting_cash=resulting_cash,
            portfolio_exposure=portfolio_exposure,
            concentration_exposure=concentration_exposure,
            sector_exposure=sector_exposure,
            geographic_exposure=geographic_exposure,
            asset_class_exposure=asset_class_exposure,
            volatility_risk=volatility_risk,
            drawdown_risk=drawdown_risk,
            liquidity_risk=liquidity_risk,
            crisis_risk=crisis_risk,
            fundamental_risk=fundamental_risk,
            thesis_risk=thesis_risk,
            correlation_risk=correlation_risk,
            data_quality_risk=data_quality_risk,
            warnings=list(dict.fromkeys(warnings)),
            violated_rules=violated,
            passed_rules=passed,
            required_actions=list(dict.fromkeys(required_actions)),
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            recommended_max_position_size=max(0.0, max_position_value),
            metadata={
                "hard_violation_count": len(hard_violations),
                "data_quality_violation_count": len(insufficient_rules),
                "soft_warning_count": len(warnings),
                "resulting_cash_reserve_pct": cash_reserve_pct,
            },
        )

    def _classified_exposure(
        self,
        portfolio: PortfolioRiskState,
        proposal: InvestmentProposal,
        attr: str,
        limit: float,
        dimension: RiskDimension,
        rule_id: str,
        passed: List[RiskRuleResult],
        violated: List[RiskRuleResult],
        required_actions: List[str],
    ) -> Optional[float]:
        value = getattr(proposal, attr)
        if not value:
            required_actions.append(f"Provide {attr} classification for diversification risk.")
            return None
        existing = self._sum_exposure(portfolio.positions, attr, value)
        if existing is None:
            required_actions.append(f"Existing position {attr} classifications are unavailable.")
            return None
        exposure = (existing + proposal.proposed_position_value) / portfolio.portfolio_value
        self._add_rule(passed, violated, RiskRuleResult(
            rule_id=rule_id,
            dimension=dimension,
            rule_type=RiskRuleType.HARD,
            passed=exposure <= limit,
            description=f"Resulting {attr} exposure must stay within configured limit.",
            observed_value=exposure,
            limit_value=limit,
            severity=RiskSeverity.HIGH,
            evidence_ids=["portfolio_state", "risk_policy"],
        ))
        return exposure

    def _volatility_check(self, snapshot, policy, passed, violated, warnings, required_actions, now) -> RiskSeverity:
        if snapshot is None:
            if policy.require_market_data:
                self._add_rule(passed, violated, RiskRuleResult(
                    rule_id="required_market_data",
                    dimension=RiskDimension.DATA_QUALITY,
                    rule_type=RiskRuleType.HARD,
                    passed=False,
                    description="Market snapshot is required by policy.",
                    severity=RiskSeverity.CRITICAL,
                ))
                required_actions.append("Provide current Market Research output.")
            return RiskSeverity.UNKNOWN
        retrieved_at = snapshot.retrieved_at or snapshot.timestamp
        if retrieved_at and now - retrieved_at > timedelta(minutes=policy.stale_market_data_minutes):
            warnings.append("Market data is stale relative to risk policy.")
            return RiskSeverity.MODERATE
        vol = snapshot.volatility.short_term_volatility
        if vol is None:
            warnings.append("Volatility is unavailable in market snapshot.")
            return RiskSeverity.UNKNOWN
        if vol > policy.volatility_block_threshold:
            self._add_rule(passed, violated, RiskRuleResult(
                rule_id="max_volatility_threshold",
                dimension=RiskDimension.VOLATILITY,
                rule_type=RiskRuleType.HARD,
                passed=False,
                description="Observed volatility exceeds hard block threshold.",
                observed_value=vol,
                limit_value=policy.volatility_block_threshold,
                severity=RiskSeverity.HIGH,
                evidence_ids=["market_snapshot"],
            ))
            return RiskSeverity.HIGH
        if vol > policy.volatility_warning_threshold:
            warnings.append("Observed volatility exceeds warning threshold.")
            return RiskSeverity.MODERATE
        return RiskSeverity.LOW

    def _liquidity_check(self, snapshot, warnings) -> RiskSeverity:
        if snapshot is None:
            return RiskSeverity.UNKNOWN
        if snapshot.volume.current_volume <= 0:
            warnings.append("Current volume is zero or unavailable.")
            return RiskSeverity.HIGH
        if snapshot.volume.is_unusual_volume:
            warnings.append("Unusual volume may affect liquidity assessment.")
            return RiskSeverity.MODERATE
        return RiskSeverity.LOW

    def _crisis_check(self, crisis_result, policy, passed, violated, warnings) -> RiskSeverity:
        if not isinstance(crisis_result, AgentResult):
            return RiskSeverity.UNKNOWN
        severities = []
        for event in crisis_result.analysis.get("events", []) or []:
            if isinstance(event, dict):
                severities.append(self._severity_from_text(event.get("severity")))
                risk_dims = (event.get("risk") or {}).get("dimensions") or {}
                severities.append(self._severity_from_text(risk_dims.get("overall_risk_level")))
        highest = max(severities, key=lambda s: _SEVERITY_RANK[s], default=RiskSeverity.UNKNOWN)
        if policy.block_on_severe_crisis and _SEVERITY_RANK[highest] >= _SEVERITY_RANK[policy.crisis_block_severity]:
            self._add_rule(passed, violated, RiskRuleResult(
                rule_id="crisis_block_severity",
                dimension=RiskDimension.CRISIS,
                rule_type=RiskRuleType.HARD,
                passed=False,
                description="Crisis exposure meets or exceeds configured block severity.",
                severity=RiskSeverity.CRITICAL,
                evidence_ids=["crisis_research_result"],
            ))
        elif _SEVERITY_RANK[highest] >= _SEVERITY_RANK[policy.crisis_warning_severity]:
            warnings.append("Crisis/geopolitical risk meets warning severity.")
        return highest

    def _deep_research_check(self, dossier, policy, passed, violated, warnings, required_actions):
        if dossier is None:
            if policy.require_deep_research:
                self._add_rule(passed, violated, RiskRuleResult(
                    rule_id="required_deep_research",
                    dimension=RiskDimension.DATA_QUALITY,
                    rule_type=RiskRuleType.HARD,
                    passed=False,
                    description="Deep Looker dossier is required by policy.",
                    severity=RiskSeverity.CRITICAL,
                ))
                required_actions.append("Provide Deep Looker dossier.")
                return RiskSeverity.UNKNOWN, RiskSeverity.UNKNOWN, RiskSeverity.CRITICAL
            return RiskSeverity.UNKNOWN, RiskSeverity.UNKNOWN, RiskSeverity.UNKNOWN
        data_risk = RiskSeverity.LOW
        if dossier.confidence < policy.min_data_confidence:
            self._add_rule(passed, violated, RiskRuleResult(
                rule_id="minimum_data_confidence",
                dimension=RiskDimension.DATA_QUALITY,
                rule_type=RiskRuleType.HARD,
                passed=False,
                description="Deep research confidence is below policy minimum.",
                observed_value=dossier.confidence,
                limit_value=policy.min_data_confidence,
                severity=RiskSeverity.CRITICAL,
                evidence_ids=["deep_research_dossier"],
            ))
            data_risk = RiskSeverity.CRITICAL
        if dossier.contradictions:
            warnings.append("Deep Looker reported unresolved contradictions.")
            data_risk = max(data_risk, RiskSeverity.MODERATE, key=lambda s: _SEVERITY_RANK[s])
        fundamental = RiskSeverity.MODERATE if dossier.unknowns else RiskSeverity.LOW
        thesis_status = dossier.thesis.status.value
        thesis = RiskSeverity.HIGH if thesis_status in ("WEAK_SUPPORT", "INSUFFICIENT_DATA") else RiskSeverity.MODERATE if thesis_status == "MIXED" else RiskSeverity.LOW
        if _SEVERITY_RANK[thesis] >= _SEVERITY_RANK[RiskSeverity.MODERATE]:
            warnings.append(f"Deep Looker thesis status is {thesis_status}.")
        return fundamental, thesis, data_risk

    def _correlation_check(self, portfolio, proposal, policy, warnings) -> RiskSeverity:
        group = proposal.metadata.get("correlation_group")
        if not group:
            return RiskSeverity.UNKNOWN
        exposure = sum(p.market_value for p in portfolio.positions if p.correlation_group == group)
        correlated_pct = (exposure + proposal.proposed_position_value) / portfolio.portfolio_value
        if correlated_pct > policy.max_correlated_exposure_pct:
            warnings.append("Correlated exposure exceeds warning threshold.")
            return RiskSeverity.MODERATE
        return RiskSeverity.LOW

    def _news_check(self, news_result, warnings) -> None:
        if not isinstance(news_result, AgentResult):
            return
        direction = str((news_result.impact or {}).get("direction", "")).upper()
        if direction == "NEGATIVE":
            warnings.append("News Research reports negative impact direction.")
        for warning in news_result.warnings:
            if "conflict" in warning.lower() or "regulat" in warning.lower() or "legal" in warning.lower():
                warnings.append(f"News risk warning: {warning}")

    def _confidence(self, policy, snapshot, dossier, crisis_result, violated) -> float:
        score = 0.85
        if policy.require_market_data and snapshot is None:
            score -= 0.25
        if policy.require_deep_research and dossier is None:
            score -= 0.25
        if dossier is not None:
            score = min(score, max(0.1, dossier.confidence))
        if crisis_result is not None:
            score = min(score, float(getattr(crisis_result, "confidence", 0.7)))
        score -= 0.05 * len(violated)
        return max(0.0, min(1.0, score))

    def _reasoning(self, decision, violated, warnings) -> str:
        if violated:
            names = ", ".join(r.rule_id for r in violated)
            return f"Deterministic risk engine set decision to {decision.value}; violated rules: {names}."
        if warnings:
            return f"Hard rules passed, but warnings require attention: {'; '.join(warnings[:3])}."
        return "All mandatory deterministic risk rules passed."

    def _llm_explanation(self, assessment: RiskAssessment) -> Dict[str, Any]:
        if self.llm_provider is None:
            return {}
        context = {
            "deterministic_decision": assessment.decision.value,
            "violated_rules": [asdict(r) for r in assessment.violated_rules],
            "warnings": assessment.warnings,
            "required_actions": assessment.required_actions,
            "position_metrics": {
                "portfolio_exposure": assessment.portfolio_exposure,
                "concentration_exposure": assessment.concentration_exposure,
                "resulting_cash": assessment.resulting_cash,
            },
        }
        raw = self.llm_provider.generate_structured(
            prompt=f"Summarize this deterministic risk assessment as JSON. Do not change the decision.\n{context}",
            schema={},
            system_prompt=self.SYSTEM_PROMPT,
        )
        clean, _ = sanitize_llm_payload(raw)
        return clean.get("analysis", {}) if isinstance(clean.get("analysis"), dict) else {}

    def process_task(self, task: Task) -> AgentResult:
        self.status = AgentStatus.RUNNING
        now = now_utc()
        try:
            proposal = self._proposal_from_input(task.input_data)
            policy = self._policy_from_input(task.input_data)
            portfolio = self._portfolio_from_input(task.input_data)
            generation_id = str(task.input_data.get("generation_id", "gen_current"))
            self._emit(RiskAssessmentStarted(
                event_id=generate_id("evt_risk_start"),
                timestamp=now,
                event_type="RiskAssessmentStarted",
                agent_id=self.agent_id,
                task_id=task.task_id,
                proposal_id=proposal.proposal_id,
                asset=proposal.asset,
            ))

            assessment = self.evaluate_investment(
                proposal=proposal,
                portfolio=portfolio,
                policy=policy,
                market_snapshot=self._market_snapshot(task.input_data),
                news_result=task.input_data.get("news_research_result"),
                crisis_result=task.input_data.get("crisis_research_result"),
                deep_dossier=self._deep_dossier(task.input_data),
                generation_id=generation_id,
            )

            llm_analysis = {}
            try:
                llm_analysis = self._llm_explanation(assessment)
            except Exception as e:
                assessment.warnings.append(f"LLM explanation unavailable: {e}")

            self._emit(RiskAssessmentCompleted(
                event_id=generate_id("evt_risk_done"),
                timestamp=now_utc(),
                event_type="RiskAssessmentCompleted",
                agent_id=self.agent_id,
                task_id=task.task_id,
                assessment_id=assessment.assessment_id,
                proposal_id=proposal.proposal_id,
                asset=proposal.asset,
                decision=assessment.decision.value,
                confidence=assessment.confidence,
                violated_rule_count=len(assessment.violated_rules),
            ))
            for rule in assessment.violated_rules:
                self._emit(RiskViolationEvent(
                    event_id=generate_id("evt_risk_violation"),
                    timestamp=now_utc(),
                    event_type="RiskViolationEvent",
                    agent_id=self.agent_id,
                    task_id=task.task_id,
                    assessment_id=assessment.assessment_id,
                    proposal_id=proposal.proposal_id,
                    asset=proposal.asset,
                    rule_id=rule.rule_id,
                    severity=rule.severity.value,
                    description=rule.description,
                ))

            if self.memory_store:
                self.memory_store.save(MemoryRecord(
                    memory_id=generate_id("mem_risk"),
                    memory_type=MemoryType.ANALYSIS,
                    generation_id=generation_id,
                    timestamp=assessment.timestamp,
                    source_agent=self.agent_id,
                    importance=9 if assessment.decision == RiskDecision.BLOCKED else 7,
                    content={
                        "assessment_id": assessment.assessment_id,
                        "proposal_id": assessment.proposal_id,
                        "asset": assessment.asset,
                        "decision": assessment.decision.value,
                        "violated_rules": [r.rule_id for r in assessment.violated_rules],
                        "warnings": assessment.warnings,
                    },
                    metadata={"assessment_id": assessment.assessment_id, "asset": assessment.asset},
                ))

            self.status = AgentStatus.SUCCESS
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                timestamp=assessment.timestamp,
                status=AgentStatus.SUCCESS,
                summary=assessment.reasoning,
                confidence=assessment.confidence,
                analysis={
                    "risk_assessment": assessment,
                    "decision": assessment.decision.value,
                    "llm_explanation": llm_analysis,
                    "execution_allowed": assessment.execution_allowed(),
                },
                impact={
                    "decision": assessment.decision.value,
                    "affected_symbols": [assessment.asset],
                    "execution_allowed": assessment.execution_allowed(),
                },
                warnings=assessment.warnings,
                metadata={
                    "assessment_id": assessment.assessment_id,
                    "proposal_id": assessment.proposal_id,
                    "asset": assessment.asset,
                    "decision": assessment.decision.value,
                },
            )
        except Exception as e:
            logger.error("RiskManagerAgent failure: %s", e)
            self.status = AgentStatus.FAILED
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                timestamp=now,
                status=AgentStatus.FAILED,
                summary="Failed to process risk assessment task.",
                confidence=0.0,
                errors=[ErrorInfo(error_code="RISK_MANAGER_FAILURE", message=str(e), details=type(e).__name__)],
            )

    def execute_order(self, *args, **kwargs):
        raise PermissionError("RiskManagerAgent cannot execute orders.")

    def modify_portfolio(self, *args, **kwargs):
        raise PermissionError("RiskManagerAgent cannot modify portfolio state.")

    def modify_capital(self, *args, **kwargs):
        raise PermissionError("RiskManagerAgent cannot modify capital.")

    def change_strategy(self, *args, **kwargs):
        raise PermissionError("RiskManagerAgent cannot alter investment strategies.")

    def approve_investment(self, *args, **kwargs):
        raise PermissionError("RiskManagerAgent cannot approve investments; it only assesses risk.")
