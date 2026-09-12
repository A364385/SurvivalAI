from typing import List, Optional, Dict, Any
from app.agents.base import BaseAgent
from app.core.models.agent import AgentConfig, AgentStatus, AgentResult
from app.core.models.task import Task
from app.core.models.knowledge import Fact, Source
from app.core.models.error import ErrorInfo
from app.core.models.memory import MemoryRecord, MemoryType
from app.core.models.events import NewsEvent
from app.core.memory.store import MemoryStore
from app.services.news.provider import NewsProvider
from app.services.llm.provider import LLMProvider
from app.agents.news_research.deduplicator import NewsDeduplicator
from app.agents.news_research.conflict_detector import ConflictDetector
from app.agents.news_research.recency_tracker import RecencyTracker
from app.agents.news_research.prompt_builder import NewsPromptBuilder
from app.utils.ids import generate_id
from app.utils.time import now_utc
from app.utils.logging import get_logger

logger = get_logger(__name__)

class NewsResearchAgent(BaseAgent):
    """Autonomous research agent responsible for market news gathering,
    source validation, deduplication, conflict detection, and LLM fact extraction.
    
    STRICT SAFETY:
    This is a research-only agent. It has NO order execution or portfolio control logic.
    """

    def __init__(
        self,
        agent_id: str,
        news_provider: NewsProvider,
        llm_provider: LLMProvider,
        memory_store: Optional[MemoryStore] = None,
        configuration: Optional[AgentConfig] = None
    ):
        config = configuration or AgentConfig(version="1.0.0", model="gemini/claude")
        super().__init__(
            agent_id=agent_id,
            agent_name="NewsResearchAgent",
            role="Market News & Qualitative Intelligence Researcher",
            description="Ingests real-world news feeds, validates sources, deduplicates reports, and produces structured fact-based intelligence.",
            version="1.0.0",
            configuration=config,
            capabilities=["news_retrieval", "source_validation", "deduplication", "event_clustering", "fact_extraction", "impact_analysis"]
        )
        self.news_provider = news_provider
        self.llm_provider = llm_provider
        self.memory_store = memory_store

        self.deduplicator = NewsDeduplicator()
        self.conflict_detector = ConflictDetector()
        self.recency_tracker = RecencyTracker()
        self.prompt_builder = NewsPromptBuilder()

    def process_task(self, task: Task) -> AgentResult:
        """Executes news intelligence task.
        Extracts symbol or general query, clusters incoming items, calls LLM,
        validates output, updates memory, and returns typed AgentResult.
        """
        self.status = AgentStatus.RUNNING
        now = now_utc()
        target_symbol = task.input_data.get("symbol")
        limit = task.input_data.get("limit", 20)

        logger.info(f"NewsResearchAgent processing task {task.task_id} for symbol={target_symbol}")

        try:
            # 1. Retrieval
            if target_symbol:
                raw_news = self.news_provider.get_news_for_symbol(target_symbol, limit=limit)
            else:
                raw_news = self.news_provider.get_latest_news(limit=limit)

            if not raw_news:
                self.status = AgentStatus.SUCCESS
                return AgentResult(
                    agent_id=self.agent_id,
                    task_id=task.task_id,
                    timestamp=now,
                    status=AgentStatus.SUCCESS,
                    summary=f"No recent news found for query (symbol={target_symbol}).",
                    confidence=1.0,
                    warnings=["No news articles matched the criteria."]
                )

            # 2. Deduplication & Event Clustering
            clusters = self.deduplicator.cluster_items(raw_news)

            # 3. Conflict & Recency analysis on top cluster
            primary_cluster = clusters[0]
            primary_cluster = self.conflict_detector.detect_conflicts(primary_cluster)
            primary_cluster = self.recency_tracker.evaluate_recency(primary_cluster, now)

            # 4. LLM Analysis with secure prompt separation
            prompt = self.prompt_builder.build_analysis_prompt(primary_cluster)
            llm_result = self.llm_provider.generate_structured(
                prompt=prompt,
                schema={},
                system_prompt=self.prompt_builder.SYSTEM_PROMPT
            )

            # 5. Extract and normalize Facts
            facts: List[Fact] = []
            for f in llm_result.get("facts", []):
                src_id = f.get("source_id")
                facts.append(Fact(
                    fact_id=generate_id("fact"),
                    statement=f.get("statement", ""),
                    source_ids=[src_id] if src_id else [s.source_id for s in primary_cluster.sources],
                    timestamp=now,
                    confidence=float(f.get("confidence", primary_cluster.confidence)),
                    data_type=f.get("data_type", "news_fact")
                ))

            # 6. Warnings
            warnings: List[str] = list(llm_result.get("warnings", []))
            if primary_cluster.has_conflicts:
                warnings.extend(primary_cluster.conflicting_claims)
            if primary_cluster.is_stale:
                warnings.append("Cluster information may be stale (>7 days).")
            if primary_cluster.superseded_by:
                warnings.append(primary_cluster.superseded_by)

            # 7. Construct AgentResult
            result = AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                timestamp=now,
                status=AgentStatus.SUCCESS,
                summary=primary_cluster.summary,
                confidence=float(llm_result.get("confidence", primary_cluster.confidence)),
                facts=facts,
                analysis=llm_result.get("analysis", {}),
                impact=llm_result.get("impact", {}),
                warnings=warnings,
                sources=primary_cluster.sources,
                metadata={
                    "cluster_id": primary_cluster.cluster_id,
                    "event_type": primary_cluster.event_type.value,
                    "total_clusters": len(clusters),
                    "articles_in_cluster": len(primary_cluster.articles)
                }
            )

            # 8. Memory Store Integration
            if self.memory_store:
                # Store analysis record
                gen_id = task.input_data.get("generation_id", "gen_current")
                self.memory_store.save(MemoryRecord(
                    memory_id=generate_id("mem_news"),
                    memory_type=MemoryType.ANALYSIS,
                    generation_id=gen_id,
                    timestamp=now,
                    source_agent=self.agent_id,
                    importance=int(primary_cluster.importance * 10),
                    content={
                        "summary": result.summary,
                        "facts": [f.statement for f in facts],
                        "impact": result.impact,
                        "cluster_id": primary_cluster.cluster_id
                    },
                    metadata=result.metadata
                ))

            self.status = AgentStatus.SUCCESS
            return result

        except Exception as e:
            logger.error(f"Error in NewsResearchAgent: {str(e)}")
            self.status = AgentStatus.FAILED
            err = ErrorInfo(
                error_code="NEWS_RESEARCH_FAILURE",
                message=str(e),
                details=type(e).__name__
            )
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                timestamp=now,
                status=AgentStatus.FAILED,
                summary="Failed to process news research task.",
                confidence=0.0,
                errors=[err]
            )

    # STRICT PERMISSION ENFORCEMENT:
    # Explicitly prohibit trading/order methods on research agent
    def execute_order(self, *args, **kwargs):
        raise PermissionError("NewsResearchAgent is a research-only agent and cannot execute orders.")

    def modify_portfolio(self, *args, **kwargs):
        raise PermissionError("NewsResearchAgent cannot modify portfolio state.")

    def change_strategy(self, *args, **kwargs):
        raise PermissionError("NewsResearchAgent cannot alter investment strategies.")
