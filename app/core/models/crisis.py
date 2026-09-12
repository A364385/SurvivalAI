from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from app.core.models.knowledge import Evidence, Source


class CrisisEventType(str, Enum):
    WAR = "WAR"
    MILITARY_CONFLICT = "MILITARY_CONFLICT"
    TERRORISM_EVENT = "TERRORISM_EVENT"
    POLITICAL_INSTABILITY = "POLITICAL_INSTABILITY"
    ELECTION = "ELECTION"
    GOVERNMENT_CHANGE = "GOVERNMENT_CHANGE"
    DIPLOMATIC_CRISIS = "DIPLOMATIC_CRISIS"
    SANCTIONS = "SANCTIONS"
    TRADE_RESTRICTION = "TRADE_RESTRICTION"
    TARIFF = "TARIFF"
    EXPORT_CONTROL = "EXPORT_CONTROL"
    ENERGY_DISRUPTION = "ENERGY_DISRUPTION"
    SUPPLY_CHAIN_DISRUPTION = "SUPPLY_CHAIN_DISRUPTION"
    CENTRAL_BANK = "CENTRAL_BANK"
    INTEREST_RATE = "INTEREST_RATE"
    INFLATION = "INFLATION"
    RECESSION_RISK = "RECESSION_RISK"
    DEBT_CRISIS = "DEBT_CRISIS"
    CURRENCY_CRISIS = "CURRENCY_CRISIS"
    REGULATION = "REGULATION"
    NATURAL_DISASTER = "NATURAL_DISASTER"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
    CYBER_INCIDENT = "CYBER_INCIDENT"
    OTHER = "OTHER"


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class EscalationStatus(str, Enum):
    ESCALATING = "ESCALATING"
    STABLE = "STABLE"
    DE_ESCALATING = "DE_ESCALATING"
    RESOLVED = "RESOLVED"
    UNKNOWN = "UNKNOWN"


class GeographicScope(str, Enum):
    LOCAL = "LOCAL"
    NATIONAL = "NATIONAL"
    REGIONAL = "REGIONAL"
    MULTI_COUNTRY = "MULTI_COUNTRY"
    GLOBAL = "GLOBAL"
    UNKNOWN = "UNKNOWN"


class CrisisTimeHorizon(str, Enum):
    IMMEDIATE = "IMMEDIATE"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"
    UNKNOWN = "UNKNOWN"


class TransmissionChannel(str, Enum):
    ENERGY_PRICES = "ENERGY_PRICES"
    COMMODITY_PRICES = "COMMODITY_PRICES"
    INFLATION = "INFLATION"
    INTEREST_RATES = "INTEREST_RATES"
    CURRENCY = "CURRENCY"
    TRADE = "TRADE"
    SUPPLY_CHAINS = "SUPPLY_CHAINS"
    CONSUMER_DEMAND = "CONSUMER_DEMAND"
    PRODUCTION_COSTS = "PRODUCTION_COSTS"
    TRANSPORTATION = "TRANSPORTATION"
    FINANCING_CONDITIONS = "FINANCING_CONDITIONS"
    REGULATION = "REGULATION"
    INVESTOR_RISK_PERCEPTION = "INVESTOR_RISK_PERCEPTION"


class CrisisImpactDirection(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    MIXED = "MIXED"
    UNCERTAIN = "UNCERTAIN"
    NO_MATERIAL_IMPACT_IDENTIFIED = "NO_MATERIAL_IMPACT_IDENTIFIED"


class ImpactMagnitude(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"
    UNKNOWN = "UNKNOWN"


class ClaimKind(str, Enum):
    CONFIRMED_FACT = "CONFIRMED_FACT"
    REPORTED_CLAIM = "REPORTED_CLAIM"
    ANALYTICAL_INTERPRETATION = "ANALYTICAL_INTERPRETATION"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class ProbabilityLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    UNCERTAIN = "UNCERTAIN"
    UNKNOWN = "UNKNOWN"


class ExposureCategory(str, Enum):
    COUNTRY = "COUNTRY"
    REGION = "REGION"
    INDUSTRY = "INDUSTRY"
    SECTOR = "SECTOR"
    COMPANY = "COMPANY"
    COMMODITY = "COMMODITY"
    CURRENCY = "CURRENCY"
    ASSET_CLASS = "ASSET_CLASS"


SEVERITY_RANK = {
    SeverityLevel.UNKNOWN: 0,
    SeverityLevel.LOW: 1,
    SeverityLevel.MODERATE: 2,
    SeverityLevel.HIGH: 3,
    SeverityLevel.SEVERE: 4,
    SeverityLevel.CRITICAL: 5,
}

RISK_RANK = {
    RiskLevel.UNKNOWN: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MODERATE: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.SEVERE: 4,
    RiskLevel.CRITICAL: 5,
}


@dataclass
class ClaimRecord:
    """A statement labeled as confirmed fact, reported claim, or interpretation."""
    statement: str
    kind: ClaimKind
    source_ids: List[str]
    confidence: float


@dataclass
class TransmissionStep:
    """One hop in a market-transmission path. Not a causal proof."""
    channel: TransmissionChannel
    description: str
    confidence: float = 0.5


@dataclass
class TransmissionPath:
    steps: List[TransmissionStep] = field(default_factory=list)
    narrative: str = ""


@dataclass
class ExposureItem:
    category: ExposureCategory
    name: str
    reason: str
    impact_direction: CrisisImpactDirection = CrisisImpactDirection.UNCERTAIN
    impact_magnitude: ImpactMagnitude = ImpactMagnitude.UNKNOWN


@dataclass
class RiskDimensions:
    """Inspectable risk factors for a downstream Risk Manager. Not a single opaque score."""
    geopolitical_risk: RiskLevel = RiskLevel.UNKNOWN
    political_risk: RiskLevel = RiskLevel.UNKNOWN
    trade_risk: RiskLevel = RiskLevel.UNKNOWN
    energy_risk: RiskLevel = RiskLevel.UNKNOWN
    supply_chain_risk: RiskLevel = RiskLevel.UNKNOWN
    macroeconomic_risk: RiskLevel = RiskLevel.UNKNOWN
    regulatory_risk: RiskLevel = RiskLevel.UNKNOWN
    overall_risk_level: RiskLevel = RiskLevel.UNKNOWN
    overall_method: str = "MAX_OF_DECLARED_DIMENSIONS"


@dataclass
class RiskFactors:
    """Separated risk inputs. Downstream systems may aggregate; this agent does not invent 'Risk=83'."""
    severity: SeverityLevel = SeverityLevel.UNKNOWN
    probability: ProbabilityLevel = ProbabilityLevel.UNCERTAIN
    exposure: ImpactMagnitude = ImpactMagnitude.UNKNOWN
    horizon: CrisisTimeHorizon = CrisisTimeHorizon.UNKNOWN
    dimensions: RiskDimensions = field(default_factory=RiskDimensions)


@dataclass
class TimelineEntry:
    label: str
    description: str
    occurred_at: Optional[datetime] = None
    source_ids: List[str] = field(default_factory=list)


@dataclass
class CrisisLink:
    from_event_id: str
    to_event_id: str
    relationship: str
    rationale: str
    confidence: float


@dataclass
class CrisisUpdate:
    previous_event_id: str
    change_summary: str
    prior_severity: SeverityLevel
    new_severity: SeverityLevel
    prior_escalation: EscalationStatus
    new_escalation: EscalationStatus
    changed_fields: List[str] = field(default_factory=list)


@dataclass
class GeopoliticalCrisis:
    """Structured geopolitical / macro risk intelligence. Not an investment decision."""
    event_id: str
    detected_at: datetime
    event_timestamp: Optional[datetime]
    event_type: CrisisEventType
    title: str
    summary: str
    countries_involved: List[str] = field(default_factory=list)
    regions_affected: List[str] = field(default_factory=list)
    entities_affected: List[str] = field(default_factory=list)
    sectors_affected: List[str] = field(default_factory=list)
    assets_affected: List[str] = field(default_factory=list)
    commodities_affected: List[str] = field(default_factory=list)
    currencies_affected: List[str] = field(default_factory=list)
    severity: SeverityLevel = SeverityLevel.UNKNOWN
    escalation_status: EscalationStatus = EscalationStatus.UNKNOWN
    geographic_scope: GeographicScope = GeographicScope.UNKNOWN
    time_horizon: CrisisTimeHorizon = CrisisTimeHorizon.UNKNOWN
    confidence: float = 0.0
    sources: List[Source] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    claims: List[ClaimRecord] = field(default_factory=list)
    secondary_types: List[CrisisEventType] = field(default_factory=list)
    transmission: TransmissionPath = field(default_factory=TransmissionPath)
    exposures: List[ExposureItem] = field(default_factory=list)
    impact_direction: CrisisImpactDirection = CrisisImpactDirection.UNCERTAIN
    impact_magnitude: ImpactMagnitude = ImpactMagnitude.UNKNOWN
    risk: RiskFactors = field(default_factory=RiskFactors)
    timeline: List[TimelineEntry] = field(default_factory=list)
    links: List[CrisisLink] = field(default_factory=list)
    update: Optional[CrisisUpdate] = None
    uncertainty: List[str] = field(default_factory=list)
    conflicting_claims: List[str] = field(default_factory=list)
    market_correlation_note: Optional[str] = None
    cluster_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
