from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.agents.base import BaseAgent
from app.core.models.memory import MemoryRecord, MemoryType
from app.core.models.generation import DeathReport, Experience
from app.core.models.knowledge import Knowledge
from app.core.models.events import BaseEvent
from app.utils.ids import generate_id
from app.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class LossAnalysisReport:
    generation_id: str
    root_cause: str
    failed_actions: List[str]
    successful_actions: List[str]
    market_conditions: Dict[str, Any]
    lessons_learned: List[str]
    confidence: float
    severity_score: float

class LossAnalyzerAgent(BaseAgent):
    """The Forensic Scientist: Analyzes 'Deaths' to extract lessons for future generations."""

    def __init__(
        self,
        agent_id: str,
        llm_provider: Optional[Any] = None,
        memory_store: Optional[Any] = None,
        configuration: Optional[Any] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_name="LossAnalyzerAgent",
            role="Forensic Loss Analyst",
            description="Analyzes failed generations to extract validated lessons for the evolutionary loop.",
            version="1.0.0",
            configuration=configuration,
        )
        self.llm_provider = llm_provider
        self.memory_store = memory_store

    def analyze_death(self, death_report: DeathReport) -> LossAnalysisReport:
        """Performs a deep forensic analysis of a generation's failure."""
        logger.info(f"Analyzing death of generation {death_report.generation_id}")

        # 1. Prepare Context (Market, Actions, Results)
        context = {
            "death_report": death_report,
            "initial_capital": death_report.starting_capital,
            "final_capital": death_report.final_capital,
            "total_return": death_report.total_return,
            "maximum_drawdown": death_report.maximum_drawdown,
            "actions": death_report.major_decisions + death_report.failed_decisions,
            "market_conditions": death_report.market_conditions,
            "suspected_causes": death_report.suspected_causes,
            "lessons_pre_analysis": death_report.lessons_for_future_analysis
        }

        # 2. LLM Analysis (The "Brain" Thinking)
        # Note: We use a specific prompt to ensure the LLM doesn't just summarize,
        # but identifies the specific logic failure.
        prompt = f"""
        You are a Forensic Financial Analyst. Analyze the following death report of an autonomous trading generation.
        
        REPORT:
        {context}

        TASK:
        1. Identify the PRIMARY ROOT CAUSE of the capital loss. (e.g., "Failure to recognize volatility," "Over-leverage," "Sentiment Bias").
        2. Identify specific actions that contributed to the failure.
        3. Identify any actions that were correct but overwhelmed by market conditions.
        4. Formulate 3-5 CONCRETE LESSONS that the successor generation should follow.
        
        OUTPUT FORMAT (JSON):
        {{
            "root_cause": "string",
            "failed_actions": ["list"],
            "successful_actions": ["list"],
            "market_conditions_summary": "string",
            "lessons_learned": ["list"],
            "confidence": float,
            "severity_score": float (0.0-1.0)
        }}
        """
        
        # Call the LLM (Logic to be integrated with the actual provider in runtime)
        # For now, we assume the result is parsed into a LossAnalysisReport.
        # In a real run, this would use self.llm_provider.generate_structured
        
        # Placeholder for the result of the LLM call
        analysis = {
            "root_cause": "Excessive exposure to high-volatility crypto assets during a geopolitical crisis.",
            "failed_actions": ["High-leverage BTC buy", "Delayed exit on news"],
            "successful_actions": ["Initial diversification"],
            "market_conditions_summary": "High volatility, major news on sanctions.",
            "lessons_learned": [
                "Reduce position size when volatility > 5% in 24h.",
                "Prioritize news on sanctions over technical indicators.",
                "Maintain higher cash reserve during geopolitical tension."
            ],
            "confidence": 0.92,
            "severity_score": 0.85
        }

        return LossAnalysisReport(
            generation_id=death_report.generation_id,
            root_cause=analysis["root_cause"],
            failed_actions=analysis["failed_actions"],
            successful_actions=analysis["successful_actions"],
            market_conditions=analysis["market_conditions_summary"],
            lessons_learned=analysis["lessons_learned"],
            confidence=analysis["confidence"],
            severity_score=analysis["severity_score"]
        )

    def _emit_analysis(self, report: LossAnalysisReport) -> None:
        # Logic to save the report to MemoryStore and publish the event
        pass
