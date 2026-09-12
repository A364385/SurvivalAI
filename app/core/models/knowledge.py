from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

class SourceType(str, Enum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    TERTIARY = "TERTIARY"

@dataclass
class Source:
    source_id: str
    title: str
    publisher: str
    url: str
    published_at: Optional[datetime] = None
    retrieved_at: Optional[datetime] = None
    source_type: SourceType = SourceType.SECONDARY
    is_primary: bool = False
    reliability_score: float = 0.8

@dataclass
class Evidence:
    evidence_id: str
    description: str
    source_id: str
    relevance_score: float

@dataclass
class Fact:
    fact_id: str
    statement: str
    source_ids: List[str]
    timestamp: datetime
    confidence: float
    data_type: str
