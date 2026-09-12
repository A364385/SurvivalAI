from datetime import datetime, timedelta
from app.core.models.news import NewsEventCluster

SUPERSEDING_KEYWORDS = ["clarifies", "corrects", "refutes", "denies", "updates", "revises", "walks back"]

class RecencyTracker:
    """Analyzes chronological evolution and flags when older reporting is superseded
    or made stale by subsequent official announcements or revisions.
    """

    def evaluate_recency(self, cluster: NewsEventCluster, reference_time: datetime) -> NewsEventCluster:
        articles = sorted(cluster.articles, key=lambda a: a.timestamp)
        if not articles:
            return cluster

        latest_item = articles[-1]
        latest_text = (latest_item.headline + " " + latest_item.summary).lower()

        # Check if the latest item explicitly corrects or clarifies earlier items
        if len(articles) > 1:
            for kw in SUPERSEDING_KEYWORDS:
                if kw in latest_text:
                    cluster.superseded_by = f"Superseded by update: '{latest_item.headline}' ({latest_item.source})"
                    break

        # Check staleness: if older than 7 days
        age = reference_time - cluster.last_updated
        if age > timedelta(days=7):
            cluster.is_stale = True

        return cluster
