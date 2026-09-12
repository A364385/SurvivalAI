import re
from typing import List, Dict
from datetime import datetime, timedelta
from app.core.models.news import NewsItem, NewsEventCluster, NewsEventType
from app.agents.news_research.validator import SourceValidator
from app.utils.ids import generate_id

def tokenize_headline(headline: str) -> set[str]:
    words = re.findall(r'[a-zA-Z0-9]+', headline.lower())
    stopwords = {"a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or", "is", "are", "with", "as", "from", "by", "its", "across"}
    tokens = set()
    for w in words:
        if w not in stopwords and len(w) >= 2:
            stem = w.rstrip('s') if len(w) > 3 and not w.endswith("ss") else w
            tokens.add(stem)
    return tokens

def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

class NewsDeduplicator:
    """Detects duplicate or near-duplicate news reports across different media outlets
    and groups them into structured NewsEventClusters while preserving all source citations.
    """

    def __init__(self, similarity_threshold: float = 0.25, time_window_hours: int = 48):
        self.similarity_threshold = similarity_threshold
        self.time_window = timedelta(hours=time_window_hours)
        self.validator = SourceValidator()

    def cluster_items(self, items: List[NewsItem]) -> List[NewsEventCluster]:
        if not items:
            return []

        clusters: List[NewsEventCluster] = []

        # Sort items chronologically
        sorted_items = sorted(items, key=lambda x: x.timestamp)

        for item in sorted_items:
            tokens = tokenize_headline(item.headline)
            assigned_cluster: NewsEventCluster | None = None

            for cluster in clusters:
                # Check time window
                time_diff = abs(item.timestamp - cluster.first_seen)
                if time_diff > self.time_window:
                    continue

                # Check similarity against all headlines in cluster
                for headline in cluster.headlines:
                    c_tokens = tokenize_headline(headline)
                    sim = jaccard_similarity(tokens, c_tokens)
                    shared_words = tokens & c_tokens
                    common_symbols = set(item.related_symbols) & set(cluster.affected_symbols)

                    # Cluster if high Jaccard or shared symbol with 2+ shared keywords or 3+ shared keywords
                    if sim >= self.similarity_threshold or (common_symbols and (sim >= 0.15 or len(shared_words) >= 2)) or len(shared_words) >= 3:
                        assigned_cluster = cluster
                        break
                if assigned_cluster:
                    break

            source_obj = self.validator.classify_source(item, generate_id("src"))

            if assigned_cluster:
                assigned_cluster.headlines.append(item.headline)
                assigned_cluster.sources.append(source_obj)
                assigned_cluster.articles.append(item)
                assigned_cluster.last_updated = max(assigned_cluster.last_updated, item.timestamp)
                # Combine symbols and entities
                assigned_cluster.affected_symbols = list(set(assigned_cluster.affected_symbols + item.related_symbols))
                assigned_cluster.affected_entities = list(set(assigned_cluster.affected_entities + item.related_companies))
                assigned_cluster.affected_sectors = list(set(assigned_cluster.affected_sectors + item.related_sectors))
                assigned_cluster.affected_countries = list(set(assigned_cluster.affected_countries + item.related_countries))
                # Boost confidence with multiple independent sources
                assigned_cluster.confidence = min(1.0, assigned_cluster.confidence + 0.05)
            else:
                cluster_id = generate_id("cluster")
                event_type = self._infer_event_type(item)
                new_cluster = NewsEventCluster(
                    cluster_id=cluster_id,
                    first_seen=item.timestamp,
                    last_updated=item.timestamp,
                    headlines=[item.headline],
                    sources=[source_obj],
                    articles=[item],
                    affected_entities=list(item.related_companies),
                    affected_symbols=list(item.related_symbols),
                    affected_sectors=list(item.related_sectors),
                    affected_countries=list(item.related_countries),
                    event_type=event_type,
                    importance=0.7,
                    confidence=source_obj.reliability_score,
                    summary=item.summary or item.headline
                )
                clusters.append(new_cluster)

        return clusters

    def _infer_event_type(self, item: NewsItem) -> NewsEventType:
        hl = item.headline.lower() + " " + item.summary.lower()
        if "earnings" in hl or "revenue" in hl or "quarterly" in hl or "eps" in hl:
            return NewsEventType.EARNINGS
        if "merger" in hl or "merge" in hl:
            return NewsEventType.MERGER
        if "acquire" in hl or "acquisition" in hl or "bought" in hl:
            return NewsEventType.ACQUISITION
        if "fed" in hl or "central bank" in hl or "rate" in hl or "fomc" in hl:
            return NewsEventType.CENTRAL_BANK
        if "inflation" in hl or "cpi" in hl:
            return NewsEventType.INFLATION
        if "sanction" in hl:
            return NewsEventType.SANCTIONS
        if "war" in hl or "military" in hl or "missile" in hl:
            return NewsEventType.WAR
        if "launch" in hl or "unveil" in hl or "announced new" in hl or "chip" in hl:
            return NewsEventType.PRODUCT
        if "regulation" in hl or "regulate" in hl or "sec" in hl or "antitrust" in hl:
            return NewsEventType.REGULATION
        return NewsEventType.OTHER
