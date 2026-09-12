from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.core.models.knowledge import Source

class NewsEventType(str, Enum):
    EARNINGS = "EARNINGS"
    MERGER = "MERGER"
    ACQUISITION = "ACQUISITION"
    PRODUCT = "PRODUCT"
    REGULATION = "REGULATION"
    GOVERNMENT = "GOVERNMENT"
    CENTRAL_BANK = "CENTRAL_BANK"
    INTEREST_RATES = "INTEREST_RATES"
    INFLATION = "INFLATION"
    EMPLOYMENT = "EMPLOYMENT"
    WAR = "WAR"
    SANCTIONS = "SANCTIONS"
    TRADE = "TRADE"
    ENERGY = "ENERGY"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    NATURAL_DISASTER = "NATURAL_DISASTER"
    CYBER_INCIDENT = "CYBER_INCIDENT"
    LEGAL = "LEGAL"
    MANAGEMENT_CHANGE = "MANAGEMENT_CHANGE"
    OTHER = "OTHER"

class ImpactDirection(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    MIXED = "MIXED"
    UNCERTAIN = "UNCERTAIN"
    NO_MATERIAL_IMPACT_IDENTIFIED = "NO_MATERIAL_IMPACT_IDENTIFIED"

class ImpactHorizon(str, Enum):
    IMMEDIATE = "IMMEDIATE"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"

@dataclass
class NewsItem:
    """Structured, normalized news article or event."""
    news_id: str
    timestamp: datetime
    headline: str
    summary: str
    source: str
    url: str
    related_symbols: List[str] = field(default_factory=list)
    related_companies: List[str] = field(default_factory=list)
    related_sectors: List[str] = field(default_factory=list)
    related_countries: List[str] = field(default_factory=list)
    language: str = "en"
    published_at: Optional[datetime] = None
    retrieved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NewsEventCluster:
    """Cluster of related news reports describing the same underlying market event."""
    cluster_id: str
    first_seen: datetime
    last_updated: datetime
    headlines: List[str]
    sources: List[Source]
    articles: List[NewsItem]
    affected_entities: List[str]
    affected_symbols: List[str]
    affected_sectors: List[str]
    affected_countries: List[str]
    event_type: NewsEventType
    importance: float
    confidence: float
    summary: str
    has_conflicts: bool = False
    conflicting_claims: List[str] = field(default_factory=list)
    is_stale: bool = False
    superseded_by: Optional[str] = None
