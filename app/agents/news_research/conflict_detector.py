from typing import List
from app.core.models.news import NewsEventCluster

CONFLICT_PAIRS = [
    ("surge", "plunge"),
    ("rise", "fall"),
    ("gain", "drop"),
    ("growth", "decline"),
    ("approved", "rejected"),
    ("beat", "miss"),
    ("record high", "record low"),
    ("positive", "negative")
]

class ConflictDetector:
    """Detects direct contradictions between multiple reporting sources on the same event."""

    def detect_conflicts(self, cluster: NewsEventCluster) -> NewsEventCluster:
        if len(cluster.articles) < 2:
            return cluster

        texts = [a.headline.lower() + " " + a.summary.lower() for a in cluster.articles]
        detected_claims = []

        for p1, p2 in CONFLICT_PAIRS:
            has_p1 = any(p1 in t for t in texts)
            has_p2 = any(p2 in t for t in texts)
            if has_p1 and has_p2:
                detected_claims.append(f"Discrepancy detected: Reports conflict on '{p1}' vs '{p2}'")

        if detected_claims:
            cluster.has_conflicts = True
            cluster.conflicting_claims = detected_claims
            # Reduce confidence due to conflict between sources
            cluster.confidence = max(0.2, cluster.confidence - 0.25)

        return cluster
