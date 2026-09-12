from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from app.agents.base import BaseAgent
from app.agents.crisis_risk.classifier import CrisisClassifier, is_crisis_relevant
from app.agents.crisis_risk.conflict import CrisisConflictDetector
from app.agents.crisis_risk.linker import CrisisLinker, CrisisRegistry
from app.agents.crisis_risk.prompt_builder import CrisisPromptBuilder
from app.agents.market_research.recommendation_guard import sanitize_llm_payload
from app.agents.news_research.deduplicator import NewsDeduplicator
from app.agents.news_research.recency_tracker import RecencyTracker
from app.core.memory.store import MemoryStore
from app.core.models.agent import AgentConfig, AgentResult, AgentStatus
from app.core.models.crisis import (
    ClaimKind, CrisisImpactDirection, GeopoliticalCrisis, SEVERITY_RANK, SeverityLevel,
)
from app.core.models.error import ErrorInfo
from app.core.models.events import BaseEvent, CrisisEvent
from app.core.models.knowledge import Fact
from app.core.models.market import MarketSnapshot
from app.core.models.memory import MemoryRecord, MemoryType
from app.core.models.news import NewsEventCluster, NewsItem
from app.core.models.provider_errors import InvalidResponseError, ProviderError
from app.core.models.task import Task
from app.services.llm.provider import LLMProvider
from app.services.llm.schema_validator import validate_news_analysis_schema
from app.services.macro.provider import MacroDataProvider
from app.services.news.provider import NewsProvider
from app.services.official.provider import OfficialDataProvider
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


def validate_crisis_analysis_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    return validate_news_analysis_schema(data)


class CrisisRiskAgent(BaseAgent):
    """Geopolitical and macro risk intelligence. Research-only; not an investment decision maker."""

    def __init__(
        self,
        agent_id: str,
        llm_provider: LLMProvider,
        news_provider: Optional[NewsProvider] = None,
        memory_store: Optional[MemoryStore] = None,
        configuration: Optional[AgentConfig] = None,
        official_provider: Optional[OfficialDataProvider] = None,
        macro_provider: Optional[MacroDataProvider] = None,
        event_publisher: Optional[Callable[[BaseEvent], None]] = None,
    ):
        config = configuration or AgentConfig(version="1.0.0", model=None)
        super().__init__(
            agent_id=agent_id,
            agent_name="CrisisRiskAgent",
            role="Crisis & Geopolitical Risk Analyst",
            description="Monitors geopolitical, political, trade, energy, supply-chain, and macro developments as risk intelligence.",
            version="1.0.0",
            configuration=config,
            capabilities=[
                "geopolitical_classification",
                "severity_assessment",
                "exposure_analysis",
                "transmission_mapping",
                "source_conflict_handling",
                "event_update_detection",
            ],
        )
        self.llm_provider = llm_provider
        self.news_provider = news_provider
        self.memory_store = memory_store
        self.official_provider = official_provider
        self.macro_provider = macro_provider
        self.event_publisher = event_publisher
        self.emitted_events: List[BaseEvent] = []
        self.deduplicator = NewsDeduplicator(similarity_threshold=0.22, time_window_hours=168)
        self.conflict_detector = CrisisConflictDetector()
        self.recency_tracker = RecencyTracker()
        self.classifier = CrisisClassifier()
        self.linker = CrisisLinker()
        self.registry = CrisisRegistry()
        self.prompt_builder = CrisisPromptBuilder()

    def _emit(self, event: BaseEvent) -> None:
        self.emitted_events.append(event)
        if self.event_publisher:
            self.event_publisher(event)

    def _collect_items(self, task: Task) -> List[NewsItem]:
        items: List[NewsItem] = list(task.input_data.get("news_items") or [])
        clusters = task.input_data.get("news_clusters") or []
        for cluster in clusters:
            if isinstance(cluster, NewsEventCluster):
                items.extend(cluster.articles)
        news_result = task.input_data.get("news_research_result")
        if isinstance(news_result, AgentResult):
            # Normalized news is preferred; AgentResult sources are citations only.
            pass
        if (
            "news_items" not in task.input_data
            and "news_clusters" not in task.input_data
            and not items
            and self.news_provider is not None
        ):
            symbol = task.input_data.get("symbol")
            limit = int(task.input_data.get("limit", 20))
            items = (
                self.news_provider.get_news_for_symbol(symbol, limit=limit)
                if symbol
                else self.news_provider.get_latest_news(limit=limit)
            )
        query = task.input_data.get("query") or task.input_data.get("symbol") or ""
        if self.official_provider is not None:
            try:
                official = self.official_provider.get_official_statements(str(query), limit=10)
                items.extend(official)
            except Exception as e:
                logger.warning("Official data unavailable: %s", e)
        return items

    def _prepare_clusters(self, items: List[NewsItem], now: datetime) -> List[NewsEventCluster]:
        clusters = self.deduplicator.cluster_items(items)
        prepared = []
        for cluster in clusters:
            cluster = self.conflict_detector.detect_conflicts(cluster)
            cluster = self.recency_tracker.evaluate_recency(cluster, now)
            prepared.append(cluster)
        return prepared

    def _market_note(self, crisis: GeopoliticalCrisis, snapshot: Optional[MarketSnapshot]) -> Optional[str]:
        if snapshot is None:
            return None
        ret = snapshot.returns.return_1d
        if ret is None:
            return (
                f"Market snapshot for {snapshot.symbol} is available; "
                "no 1-day return to compare. Causation is not established."
            )
        energy_hit = any(s.channel.value == "ENERGY_PRICES" for s in crisis.transmission.steps)
        if energy_hit and abs(ret) >= 0.02:
            return (
                f"Market movement in {snapshot.symbol} ({ret:.2%} 1d) is consistent with the reported event. "
                "This does not establish that the event caused the market movement."
            )
        return (
            f"A market snapshot for {snapshot.symbol} is temporally available alongside this event. "
            "Causation is not established."
        )

    def build_crisis_from_cluster(self, cluster: NewsEventCluster, now: datetime, snapshot: Optional[MarketSnapshot]) -> GeopoliticalCrisis:
        analysis = self.classifier.analyze_cluster(cluster, now)
        title = cluster.headlines[0] if cluster.headlines else cluster.summary
        crisis = GeopoliticalCrisis(
            event_id=generate_id("crisis"),
            detected_at=now,
            event_timestamp=cluster.first_seen,
            event_type=analysis["event_type"],
            title=title,
            summary=cluster.summary,
            countries_involved=analysis["countries"],
            regions_affected=analysis["regions"],
            entities_affected=list(cluster.affected_entities),
            sectors_affected=list(cluster.affected_sectors),
            assets_affected=list(cluster.affected_symbols),
            commodities_affected=analysis["commodities"],
            currencies_affected=analysis["currencies"],
            severity=analysis["severity"],
            escalation_status=analysis["escalation"],
            geographic_scope=analysis["scope"],
            time_horizon=analysis["horizon"],
            confidence=analysis["confidence"],
            sources=list(cluster.sources),
            evidence=analysis["evidence"],
            claims=analysis["claims"],
            secondary_types=analysis["secondary_types"],
            transmission=analysis["transmission"],
            exposures=analysis["exposures"],
            impact_direction=analysis["impact_direction"],
            impact_magnitude=analysis["impact_magnitude"],
            risk=analysis["risk"],
            timeline=analysis["timeline"],
            uncertainty=analysis["uncertainty"],
            conflicting_claims=list(cluster.conflicting_claims),
            cluster_id=cluster.cluster_id,
            metadata={"article_count": len(cluster.articles), "stale": cluster.is_stale},
        )
        crisis.market_correlation_note = self._market_note(crisis, snapshot)
        previous = self.registry.find_match(crisis)
        if previous is not None:
            crisis = self.registry.apply_update(previous, crisis)
        else:
            self.registry.remember(crisis)
        return crisis

    def process_task(self, task: Task) -> AgentResult:
        self.status = AgentStatus.RUNNING
        now = now_utc()
        snapshot = task.input_data.get("market_snapshot")
        if snapshot is not None and not isinstance(snapshot, MarketSnapshot):
            snapshot = None

        logger.info("CrisisRiskAgent processing task %s", task.task_id)
        try:
            items = self._collect_items(task)
            if not items:
                self.status = AgentStatus.SUCCESS
                return AgentResult(
                    agent_id=self.agent_id,
                    task_id=task.task_id,
                    timestamp=now,
                    status=AgentStatus.SUCCESS,
                    summary="No normalized news or official statements available for crisis analysis.",
                    confidence=1.0,
                    impact={"direction": CrisisImpactDirection.NO_MATERIAL_IMPACT_IDENTIFIED.value},
                    warnings=["Missing information: no input articles."],
                )

            clusters = self._prepare_clusters(items, now)
            crises: List[GeopoliticalCrisis] = []
            for cluster in clusters:
                crisis = self.build_crisis_from_cluster(cluster, now, snapshot)
                if is_crisis_relevant(crisis.event_type) or crisis.severity != SeverityLevel.UNKNOWN:
                    crises.append(crisis)

            relevant = [c for c in crises if is_crisis_relevant(c.event_type)]
            if not relevant:
                self.status = AgentStatus.SUCCESS
                return AgentResult(
                    agent_id=self.agent_id,
                    task_id=task.task_id,
                    timestamp=now,
                    status=AgentStatus.SUCCESS,
                    summary="No geopolitical or macro crisis event identified in the supplied reporting.",
                    confidence=0.7,
                    impact={"direction": CrisisImpactDirection.NO_MATERIAL_IMPACT_IDENTIFIED.value},
                    warnings=["No crisis-class event matched deterministic classifiers."],
                    sources=[s for c in clusters for s in c.sources],
                )

            links = self.linker.link(relevant)
            for crisis in relevant:
                crisis.links = [lk for lk in links if lk.from_event_id == crisis.event_id or lk.to_event_id == crisis.event_id]

            llm_result: Dict[str, Any] = {
                "facts": [],
                "analysis": {
                    "interpretation": relevant[0].summary,
                    "transmission_reasoning": relevant[0].transmission.narrative,
                },
                "impact": {
                    "direction": relevant[0].impact_direction.value,
                    "horizon": relevant[0].time_horizon.value,
                    "affected_entities": relevant[0].entities_affected,
                },
                "confidence": relevant[0].confidence,
                "warnings": [],
            }
            try:
                prompt = self.prompt_builder.build_analysis_prompt(relevant, snapshot)
                raw = self.llm_provider.generate_structured(
                    prompt=prompt,
                    schema={},
                    system_prompt=self.prompt_builder.SYSTEM_PROMPT,
                )
                llm_result = validate_crisis_analysis_schema(raw)
                llm_result, rec_warn = sanitize_llm_payload(llm_result)
            except (InvalidResponseError, ProviderError, RuntimeError, TypeError, ValueError) as e:
                logger.warning("LLM crisis interpretation failed: %s", e)
                rec_warn = [f"LLM interpretation unavailable: {e}"]
            warnings = list(rec_warn)
            for c in relevant:
                warnings.extend(c.uncertainty)
                warnings.extend(c.conflicting_claims)
                if c.metadata.get("stale"):
                    warnings.append(f"Stale information flagged for event {c.event_id}.")
            warnings.extend(llm_result.get("warnings") or [])

            facts: List[Fact] = []
            for c in relevant:
                for claim in c.claims:
                    facts.append(Fact(
                        fact_id=generate_id("cfact"),
                        statement=claim.statement,
                        source_ids=claim.source_ids,
                        timestamp=now,
                        confidence=claim.confidence,
                        data_type=claim.kind.value,
                    ))
            for f in llm_result.get("facts") or []:
                if not isinstance(f, dict):
                    continue
                stmt = f.get("statement", "")
                if stmt and not any(stmt == existing.statement for existing in facts):
                    facts.append(Fact(
                        fact_id=generate_id("cfact"),
                        statement=stmt,
                        source_ids=[f.get("source_id")] if f.get("source_id") else [],
                        timestamp=now,
                        confidence=float(f.get("confidence", 0.4)),
                        data_type=f.get("data_type", ClaimKind.ANALYTICAL_INTERPRETATION.value),
                    ))

            primary = relevant[0]
            analysis_out = llm_result.get("analysis") if isinstance(llm_result.get("analysis"), dict) else {}
            analysis_out = {
                **analysis_out,
                "events": [
                    {
                        "event_id": c.event_id,
                        "event_type": c.event_type.value,
                        "secondary_types": [t.value for t in c.secondary_types],
                        "severity": c.severity.value,
                        "escalation": c.escalation_status.value,
                        "scope": c.geographic_scope.value,
                        "horizon": c.time_horizon.value,
                        "confidence": c.confidence,
                        "is_update": c.update is not None,
                        "transmission": c.transmission.narrative,
                        "risk": {
                            "severity": c.risk.severity.value,
                            "probability": c.risk.probability.value,
                            "exposure": c.risk.exposure.value,
                            "horizon": c.risk.horizon.value,
                            "dimensions": {
                                "geopolitical_risk": c.risk.dimensions.geopolitical_risk.value,
                                "political_risk": c.risk.dimensions.political_risk.value,
                                "trade_risk": c.risk.dimensions.trade_risk.value,
                                "energy_risk": c.risk.dimensions.energy_risk.value,
                                "supply_chain_risk": c.risk.dimensions.supply_chain_risk.value,
                                "macroeconomic_risk": c.risk.dimensions.macroeconomic_risk.value,
                                "regulatory_risk": c.risk.dimensions.regulatory_risk.value,
                                "overall_risk_level": c.risk.dimensions.overall_risk_level.value,
                                "overall_method": c.risk.dimensions.overall_method,
                            },
                        },
                        "exposures": [
                            {"category": e.category.value, "name": e.name, "reason": e.reason}
                            for e in c.exposures
                        ],
                        "timeline": [
                            {"label": t.label, "description": t.description, "occurred_at": t.occurred_at.isoformat() if t.occurred_at else None}
                            for t in c.timeline
                        ],
                        "links": [
                            {"from": lk.from_event_id, "to": lk.to_event_id, "relationship": lk.relationship}
                            for lk in c.links
                        ],
                        "market_correlation_note": c.market_correlation_note,
                        "update": None if c.update is None else {
                            "previous_event_id": c.update.previous_event_id,
                            "change_summary": c.update.change_summary,
                            "changed_fields": c.update.changed_fields,
                        },
                    }
                    for c in relevant
                ],
            }
            impact = llm_result.get("impact") if isinstance(llm_result.get("impact"), dict) else {}
            impact = {
                **impact,
                "direction": primary.impact_direction.value,
                "magnitude": primary.impact_magnitude.value,
                "horizon": primary.time_horizon.value,
                "affected_countries": primary.countries_involved,
                "affected_sectors": primary.sectors_affected,
                "affected_assets": primary.assets_affected,
            }

            sources = []
            seen = set()
            for c in relevant:
                for s in c.sources:
                    if s.source_id not in seen:
                        sources.append(s)
                        seen.add(s.source_id)

            summary = analysis_out.get("interpretation") or primary.summary
            result = AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                timestamp=now,
                status=AgentStatus.SUCCESS,
                summary=str(summary),
                confidence=float(llm_result.get("confidence", primary.confidence)),
                facts=facts,
                analysis=analysis_out,
                impact=impact,
                warnings=list(dict.fromkeys(warnings)),
                sources=sources,
                metadata={
                    "event_count": len(relevant),
                    "primary_event_id": primary.event_id,
                    "primary_event_type": primary.event_type.value,
                    "severity": primary.severity.value,
                    "escalation": primary.escalation_status.value,
                },
            )

            for c in relevant:
                if SEVERITY_RANK[c.severity] >= SEVERITY_RANK[SeverityLevel.MODERATE]:
                    self._emit(CrisisEvent(
                        event_id=generate_id("evt_crisis"),
                        timestamp=now,
                        event_type="CrisisEvent",
                        crisis_id=c.event_id,
                        severity=SEVERITY_RANK[c.severity],
                        description=c.summary,
                        title=c.title,
                        severity_label=c.severity.value,
                        escalation_status=c.escalation_status.value,
                        geographic_scope=c.geographic_scope.value,
                        countries_involved=list(c.countries_involved),
                        is_update=c.update is not None,
                        agent_id=self.agent_id,
                        task_id=task.task_id,
                    ))

            if self.memory_store:
                gen_id = task.input_data.get("generation_id", "gen_current")
                self.memory_store.save(MemoryRecord(
                    memory_id=generate_id("mem_crisis"),
                    memory_type=MemoryType.ANALYSIS,
                    generation_id=gen_id,
                    timestamp=now,
                    source_agent=self.agent_id,
                    importance=min(10, 4 + SEVERITY_RANK[primary.severity]),
                    content={
                        "event_id": primary.event_id,
                        "event_type": primary.event_type.value,
                        "severity": primary.severity.value,
                        "summary": result.summary,
                    },
                    metadata=result.metadata,
                ))
                confirmed = [f for f in facts if f.data_type == ClaimKind.CONFIRMED_FACT.value]
                if confirmed:
                    self.memory_store.save(MemoryRecord(
                        memory_id=generate_id("mem_crisis_fact"),
                        memory_type=MemoryType.FACT,
                        generation_id=gen_id,
                        timestamp=now,
                        source_agent=self.agent_id,
                        importance=8,
                        content={"statements": [f.statement for f in confirmed[:5]]},
                        metadata={},
                    ))

            self.status = AgentStatus.SUCCESS
            return result
        except Exception as e:
            logger.error("CrisisRiskAgent failure: %s", e)
            self.status = AgentStatus.FAILED
            if self.memory_store:
                self.memory_store.save(MemoryRecord(
                    memory_id=generate_id("mem_crisis_err"),
                    memory_type=MemoryType.ERROR,
                    generation_id=task.input_data.get("generation_id", "gen_current"),
                    timestamp=now,
                    source_agent=self.agent_id,
                    importance=5,
                    content={"message": str(e)},
                    metadata={},
                ))
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                timestamp=now,
                status=AgentStatus.FAILED,
                summary="Failed to process crisis risk task.",
                confidence=0.0,
                errors=[ErrorInfo(error_code="CRISIS_RISK_FAILURE", message=str(e), details=type(e).__name__)],
            )

    def execute_order(self, *args, **kwargs):
        raise PermissionError("CrisisRiskAgent is a research-only agent and cannot execute orders.")

    def modify_portfolio(self, *args, **kwargs):
        raise PermissionError("CrisisRiskAgent cannot modify portfolio state.")

    def modify_capital(self, *args, **kwargs):
        raise PermissionError("CrisisRiskAgent cannot modify capital.")

    def change_strategy(self, *args, **kwargs):
        raise PermissionError("CrisisRiskAgent cannot alter investment strategies.")
