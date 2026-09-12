from typing import List
from app.core.models.news import NewsEventCluster
from app.agents.news_research.conflict_detector import ConflictDetector, CONFLICT_PAIRS

CRISIS_CONFLICT_PAIRS = CONFLICT_PAIRS + [
    ("escalat", "de-escalat"),
    ("impose", "lift"),
    ("invade", "withdraw"),
    ("sanctioned", "sanctions relief"),
    ("attack", "ceasefire"),
]


class CrisisConflictDetector(ConflictDetector):
    """Extends news conflict detection with geopolitical antonyms. Never silently picks a side."""

    def detect_conflicts(self, cluster: NewsEventCluster) -> NewsEventCluster:
        cluster = super().detect_conflicts(cluster)
        if len(cluster.articles) < 2:
            return cluster
        texts = [a.headline.lower() + " " + a.summary.lower() for a in cluster.articles]
        extra = []
        for p1, p2 in CRISIS_CONFLICT_PAIRS[len(CONFLICT_PAIRS):]:
            has_p1 = any(p1 in t for t in texts)
            has_p2 = any(p2 in t for t in texts)
            if has_p1 and has_p2:
                extra.append(f"Discrepancy detected: Reports conflict on '{p1}' vs '{p2}'")
        if extra:
            cluster.has_conflicts = True
            cluster.conflicting_claims = list(dict.fromkeys(cluster.conflicting_claims + extra))
            cluster.confidence = max(0.2, cluster.confidence - 0.1)
        return cluster
