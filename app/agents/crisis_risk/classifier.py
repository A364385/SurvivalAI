"""Deterministic geopolitical classification.

Severity, scope, transmission, and exposure are derived from evidence in text and
source quality. The LLM may interpret; it does not invent these values.
"""

import re
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from app.core.models.crisis import (
    ClaimKind, ClaimRecord, CrisisEventType, CrisisImpactDirection, CrisisTimeHorizon,
    EscalationStatus, ExposureCategory, ExposureItem, GeographicScope, ImpactMagnitude,
    ProbabilityLevel, RiskDimensions, RiskFactors, RiskLevel, SEVERITY_RANK,
    SeverityLevel, TimelineEntry, TransmissionChannel, TransmissionPath, TransmissionStep,
    RISK_RANK,
)
from app.core.models.knowledge import Evidence, Source, SourceType
from app.core.models.news import NewsEventCluster
from app.utils.ids import generate_id

COUNTRY_ALIASES = {
    "united states": "United States",
    "u.s.": "United States",
    "usa": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "united kingdom": "United Kingdom",
    "china": "China",
    "taiwan": "Taiwan",
    "russia": "Russia",
    "ukraine": "Ukraine",
    "iran": "Iran",
    "israel": "Israel",
    "gaza": "Gaza",
    "saudi": "Saudi Arabia",
    "saudi arabia": "Saudi Arabia",
    "germany": "Germany",
    "france": "France",
    "japan": "Japan",
    "south korea": "South Korea",
    "north korea": "North Korea",
    "india": "India",
    "brazil": "Brazil",
    "mexico": "Mexico",
    "canada": "Canada",
    "eu": "European Union",
    "european union": "European Union",
    "strait of hormuz": "Strait of Hormuz",
    "red sea": "Red Sea",
    "taiwan strait": "Taiwan Strait",
}

REGION_KEYWORDS = {
    "middle east": "Middle East",
    "europe": "Europe",
    "asia": "Asia",
    "latin america": "Latin America",
    "africa": "Africa",
    "gulf": "Persian Gulf",
}

COMMODITY_KEYWORDS = {
    "oil": "crude oil",
    "crude": "crude oil",
    "natural gas": "natural gas",
    "lng": "natural gas",
    "wheat": "wheat",
    "grain": "grains",
    "semiconductor": "semiconductors",
    "chip": "semiconductors",
    "lithium": "lithium",
    "rare earth": "rare earths",
    "copper": "copper",
}

CURRENCY_KEYWORDS = {
    "dollar": "USD",
    "usd": "USD",
    "euro": "EUR",
    "yen": "JPY",
    "yuan": "CNY",
    "renminbi": "CNY",
    "ruble": "RUB",
}

TYPE_PATTERNS: List[Tuple[CrisisEventType, Tuple[str, ...]]] = [
    (CrisisEventType.TARIFF, ("tariff",)),
    (CrisisEventType.EXPORT_CONTROL, ("export control", "export ban", "entity list")),
    (CrisisEventType.TRADE_RESTRICTION, ("trade war", "trade restriction", "import ban", "embargo")),
    (CrisisEventType.SANCTIONS, ("sanction", "ofac", "blocked entit")),
    (CrisisEventType.ENERGY_DISRUPTION, ("oil disruption", "pipeline", "energy supply", "strait of hormuz", "opec cut", "refinery outage", "gas disruption")),
    (CrisisEventType.SUPPLY_CHAIN_DISRUPTION, ("supply chain", "shipping disruption", "port blockage", "semiconductor shortage", "logistics disruption", "red sea shipping")),
    (CrisisEventType.CENTRAL_BANK, ("central bank", "federal reserve", "ecb ", "fomc")),
    (CrisisEventType.INTEREST_RATE, ("interest rate", "rate hike", "rate cut", "policy rate")),
    (CrisisEventType.INFLATION, ("inflation", "cpi", "consumer price")),
    (CrisisEventType.DEBT_CRISIS, ("sovereign debt", "default", "imf bailout")),
    (CrisisEventType.CURRENCY_CRISIS, ("currency crisis", "devaluat", "capital control", "fx crisis")),
    (CrisisEventType.RECESSION_RISK, ("recession", "economic contraction")),
    (CrisisEventType.ELECTION, ("election", "vote", "ballot")),
    (CrisisEventType.GOVERNMENT_CHANGE, ("coup", "resign", "cabinet collapse", "government overthrown")),
    (CrisisEventType.POLITICAL_INSTABILITY, ("protest", "unrest", "political crisis", "instability")),
    (CrisisEventType.DIPLOMATIC_CRISIS, ("diplomatic", "ambassador expelled", "talks collapse")),
    (CrisisEventType.REGULATION, ("regulation", "regulatory", "crypto bill", "ai act", "sec rule")),
    (CrisisEventType.NATURAL_DISASTER, ("earthquake", "hurricane", "flood", "wildfire", "tsunami")),
    (CrisisEventType.INFRASTRUCTURE_FAILURE, ("blackout", "grid failure", "infrastructure collapse")),
    (CrisisEventType.CYBER_INCIDENT, ("cyber attack", "ransomware", "data breach")),
    (CrisisEventType.TERRORISM_EVENT, ("terror", "bombing", "hostage")),
    (CrisisEventType.WAR, ("invasion", "airstrike", "missile strike", "armed conflict", " declares war", "war in ", "war with")),
    (CrisisEventType.MILITARY_CONFLICT, ("military clash", "troop buildup", "artillery", "naval clash", "skirmish")),
]

TYPE_TO_CHANNELS = {
    CrisisEventType.WAR: [
        TransmissionChannel.ENERGY_PRICES, TransmissionChannel.COMMODITY_PRICES,
        TransmissionChannel.INVESTOR_RISK_PERCEPTION, TransmissionChannel.TRADE,
    ],
    CrisisEventType.MILITARY_CONFLICT: [
        TransmissionChannel.INVESTOR_RISK_PERCEPTION, TransmissionChannel.ENERGY_PRICES,
    ],
    CrisisEventType.SANCTIONS: [
        TransmissionChannel.TRADE, TransmissionChannel.FINANCING_CONDITIONS, TransmissionChannel.CURRENCY,
    ],
    CrisisEventType.TARIFF: [TransmissionChannel.TRADE, TransmissionChannel.PRODUCTION_COSTS],
    CrisisEventType.EXPORT_CONTROL: [TransmissionChannel.SUPPLY_CHAINS, TransmissionChannel.TRADE],
    CrisisEventType.TRADE_RESTRICTION: [TransmissionChannel.TRADE, TransmissionChannel.SUPPLY_CHAINS],
    CrisisEventType.ENERGY_DISRUPTION: [
        TransmissionChannel.ENERGY_PRICES, TransmissionChannel.INFLATION, TransmissionChannel.PRODUCTION_COSTS,
    ],
    CrisisEventType.SUPPLY_CHAIN_DISRUPTION: [
        TransmissionChannel.SUPPLY_CHAINS, TransmissionChannel.PRODUCTION_COSTS, TransmissionChannel.TRANSPORTATION,
    ],
    CrisisEventType.INFLATION: [TransmissionChannel.INFLATION, TransmissionChannel.INTEREST_RATES],
    CrisisEventType.CENTRAL_BANK: [TransmissionChannel.INTEREST_RATES, TransmissionChannel.CURRENCY],
    CrisisEventType.INTEREST_RATE: [TransmissionChannel.INTEREST_RATES, TransmissionChannel.FINANCING_CONDITIONS],
    CrisisEventType.CURRENCY_CRISIS: [TransmissionChannel.CURRENCY, TransmissionChannel.FINANCING_CONDITIONS],
    CrisisEventType.DEBT_CRISIS: [TransmissionChannel.FINANCING_CONDITIONS, TransmissionChannel.CURRENCY],
    CrisisEventType.RECESSION_RISK: [TransmissionChannel.CONSUMER_DEMAND, TransmissionChannel.INVESTOR_RISK_PERCEPTION],
    CrisisEventType.REGULATION: [TransmissionChannel.REGULATION],
    CrisisEventType.CYBER_INCIDENT: [TransmissionChannel.INVESTOR_RISK_PERCEPTION],
    CrisisEventType.NATURAL_DISASTER: [TransmissionChannel.SUPPLY_CHAINS, TransmissionChannel.ENERGY_PRICES],
}

BASE_SEVERITY = {
    CrisisEventType.WAR: SeverityLevel.HIGH,
    CrisisEventType.MILITARY_CONFLICT: SeverityLevel.HIGH,
    CrisisEventType.TERRORISM_EVENT: SeverityLevel.HIGH,
    CrisisEventType.ENERGY_DISRUPTION: SeverityLevel.HIGH,
    CrisisEventType.DEBT_CRISIS: SeverityLevel.HIGH,
    CrisisEventType.CURRENCY_CRISIS: SeverityLevel.HIGH,
    CrisisEventType.SANCTIONS: SeverityLevel.MODERATE,
    CrisisEventType.EXPORT_CONTROL: SeverityLevel.MODERATE,
    CrisisEventType.TRADE_RESTRICTION: SeverityLevel.MODERATE,
    CrisisEventType.TARIFF: SeverityLevel.MODERATE,
    CrisisEventType.SUPPLY_CHAIN_DISRUPTION: SeverityLevel.MODERATE,
    CrisisEventType.CENTRAL_BANK: SeverityLevel.MODERATE,
    CrisisEventType.INTEREST_RATE: SeverityLevel.MODERATE,
    CrisisEventType.INFLATION: SeverityLevel.MODERATE,
    CrisisEventType.RECESSION_RISK: SeverityLevel.MODERATE,
    CrisisEventType.GOVERNMENT_CHANGE: SeverityLevel.MODERATE,
    CrisisEventType.POLITICAL_INSTABILITY: SeverityLevel.MODERATE,
    CrisisEventType.DIPLOMATIC_CRISIS: SeverityLevel.MODERATE,
    CrisisEventType.ELECTION: SeverityLevel.LOW,
    CrisisEventType.REGULATION: SeverityLevel.LOW,
    CrisisEventType.CYBER_INCIDENT: SeverityLevel.MODERATE,
    CrisisEventType.NATURAL_DISASTER: SeverityLevel.MODERATE,
    CrisisEventType.INFRASTRUCTURE_FAILURE: SeverityLevel.MODERATE,
    CrisisEventType.OTHER: SeverityLevel.UNKNOWN,
}

CRITICAL_KEYWORDS = ("nuclear", "invasion of", "full-scale invasion", "global energy cutoff")
SEVERE_KEYWORDS = ("blockade", "default", "strait closed", "widespread sanctions")
HIGH_KEYWORDS = ("airstrike", "major disruption", "export ban")
LOW_KEYWORDS = ("local", "minor", "routine", "scheduled")
ESCALATE_KEYWORDS = ("escalat", "expand", "additional sanction", "further strike", "worsen")
DEESCALATE_KEYWORDS = ("ceasefire", "de-escalat", "talks resume", "sanctions relief", "withdraw")
RESOLVE_KEYWORDS = ("resolved", "lifted entirely", "conflict ended", "peace treaty")


def _corpus(cluster: NewsEventCluster) -> str:
    parts = list(cluster.headlines) + [cluster.summary]
    for a in cluster.articles:
        parts.append(a.headline)
        parts.append(a.summary)
    return " ".join(parts).lower()


def extract_countries(text: str, cluster: Optional[NewsEventCluster] = None) -> List[str]:
    found = []
    lowered = text.lower()
    for alias, name in COUNTRY_ALIASES.items():
        if alias in lowered and name not in found:
            found.append(name)
    if cluster:
        for c in cluster.affected_countries:
            if c and c not in found:
                found.append(c)
    return found


def extract_named(mapping: dict, text: str) -> List[str]:
    out = []
    lowered = text.lower()
    for alias, name in mapping.items():
        if alias in lowered and name not in out:
            out.append(name)
    return out


def classify_event_types(text: str) -> Tuple[CrisisEventType, List[CrisisEventType]]:
    hits: List[CrisisEventType] = []
    lowered = text.lower()
    for etype, keys in TYPE_PATTERNS:
        if any(k in lowered for k in keys):
            hits.append(etype)
    if not hits:
        if re.search(r"\bwar\b", lowered) and "trade war" not in lowered:
            return CrisisEventType.WAR, []
        return CrisisEventType.OTHER, []
    primary = hits[0]
    secondary = [h for h in hits[1:] if h != primary]
    return primary, secondary


def classify_scope(countries: List[str], text: str) -> GeographicScope:
    lowered = text.lower()
    if any(w in lowered for w in ("global", "worldwide", "world economy")):
        return GeographicScope.GLOBAL
    if any(w in lowered for w in ("city", "municipal", "local protest", "local")) and len(countries) <= 1:
        return GeographicScope.LOCAL
    if any(w in lowered for w in ("regional", "middle east", "across europe")):
        return GeographicScope.REGIONAL
    n = len(countries)
    if n == 0:
        return GeographicScope.UNKNOWN
    if n == 1:
        return GeographicScope.NATIONAL
    if n >= 4:
        return GeographicScope.GLOBAL
    return GeographicScope.MULTI_COUNTRY


def classify_horizon(event_type: CrisisEventType) -> CrisisTimeHorizon:
    if event_type in (CrisisEventType.CENTRAL_BANK, CrisisEventType.INTEREST_RATE, CrisisEventType.TERRORISM_EVENT):
        return CrisisTimeHorizon.IMMEDIATE
    if event_type in (CrisisEventType.INFLATION, CrisisEventType.CURRENCY_CRISIS, CrisisEventType.ENERGY_DISRUPTION):
        return CrisisTimeHorizon.SHORT_TERM
    if event_type in (CrisisEventType.WAR, CrisisEventType.SANCTIONS, CrisisEventType.TRADE_RESTRICTION, CrisisEventType.REGULATION):
        return CrisisTimeHorizon.MEDIUM_TERM
    if event_type == CrisisEventType.OTHER:
        return CrisisTimeHorizon.UNKNOWN
    return CrisisTimeHorizon.UNKNOWN


def _bump(level: SeverityLevel, steps: int) -> SeverityLevel:
    rank = min(5, max(0, SEVERITY_RANK[level] + steps))
    for sev, r in SEVERITY_RANK.items():
        if r == rank:
            return sev
    return level


def classify_severity(
    event_type: CrisisEventType,
    text: str,
    sources: Sequence[Source],
    article_count: int,
) -> SeverityLevel:
    if event_type == CrisisEventType.OTHER and article_count == 0:
        return SeverityLevel.UNKNOWN
    base = BASE_SEVERITY.get(event_type, SeverityLevel.UNKNOWN)
    lowered = text.lower()
    if any(k in lowered for k in CRITICAL_KEYWORDS):
        base = _bump(base if base != SeverityLevel.UNKNOWN else SeverityLevel.HIGH, 2)
        if SEVERITY_RANK[base] < SEVERITY_RANK[SeverityLevel.CRITICAL]:
            base = SeverityLevel.CRITICAL
    elif any(k in lowered for k in SEVERE_KEYWORDS):
        base = _bump(base if base != SeverityLevel.UNKNOWN else SeverityLevel.MODERATE, 1)
        if SEVERITY_RANK[base] < SEVERITY_RANK[SeverityLevel.SEVERE]:
            base = SeverityLevel.SEVERE
    elif any(k in lowered for k in HIGH_KEYWORDS) and SEVERITY_RANK[base] < 3:
        base = SeverityLevel.HIGH
    elif any(k in lowered for k in LOW_KEYWORDS) and event_type in (CrisisEventType.ELECTION, CrisisEventType.REGULATION, CrisisEventType.POLITICAL_INSTABILITY):
        base = SeverityLevel.LOW
    # Single unverified secondary report cannot be CRITICAL.
    has_primary = any(s.is_primary or s.source_type == SourceType.PRIMARY for s in sources)
    if base == SeverityLevel.CRITICAL and not has_primary and article_count < 2:
        base = SeverityLevel.HIGH
    return base


def classify_escalation(text: str, article_count: int, has_primary: bool) -> EscalationStatus:
    lowered = text.lower()
    resolved_hits = sum(1 for k in RESOLVE_KEYWORDS if k in lowered)
    if resolved_hits and (article_count >= 2 or has_primary):
        return EscalationStatus.RESOLVED
    if resolved_hits and article_count < 2:
        return EscalationStatus.DE_ESCALATING
    if any(k in lowered for k in ESCALATE_KEYWORDS):
        return EscalationStatus.ESCALATING
    if any(k in lowered for k in DEESCALATE_KEYWORDS):
        return EscalationStatus.DE_ESCALATING
    if article_count >= 1:
        return EscalationStatus.STABLE
    return EscalationStatus.UNKNOWN


def build_transmission(event_type: CrisisEventType, secondary: List[CrisisEventType], text: str) -> TransmissionPath:
    channels: List[TransmissionChannel] = []
    for t in [event_type] + secondary:
        for ch in TYPE_TO_CHANNELS.get(t, []):
            if ch not in channels:
                channels.append(ch)
    lowered = text.lower()
    if "oil" in lowered or "energy" in lowered:
        for ch in (TransmissionChannel.ENERGY_PRICES, TransmissionChannel.INFLATION):
            if ch not in channels:
                channels.append(ch)
    if "shipping" in lowered or "port" in lowered:
        if TransmissionChannel.TRANSPORTATION not in channels:
            channels.append(TransmissionChannel.TRANSPORTATION)
    steps = [
        TransmissionStep(
            channel=ch,
            description=f"Potential transmission via {ch.value.replace('_', ' ').lower()}.",
            confidence=0.55,
        )
        for ch in channels
    ]
    narrative = " -> ".join(s.channel.value for s in steps) if steps else "No transmission path identified from available evidence."
    return TransmissionPath(steps=steps, narrative=narrative)


def build_exposures(
    event_type: CrisisEventType,
    countries: List[str],
    regions: List[str],
    cluster: NewsEventCluster,
    commodities: List[str],
    currencies: List[str],
    text: str,
) -> List[ExposureItem]:
    items: List[ExposureItem] = []
    reason = f"Potential exposure via {event_type.value.replace('_', ' ').lower()} developments."
    for c in countries:
        items.append(ExposureItem(ExposureCategory.COUNTRY, c, reason, CrisisImpactDirection.UNCERTAIN, ImpactMagnitude.MODERATE))
    for r in regions:
        items.append(ExposureItem(ExposureCategory.REGION, r, reason, CrisisImpactDirection.UNCERTAIN, ImpactMagnitude.MODERATE))
    for s in cluster.affected_sectors:
        items.append(ExposureItem(ExposureCategory.SECTOR, s, reason, CrisisImpactDirection.UNCERTAIN, ImpactMagnitude.MODERATE))
    for e in cluster.affected_entities:
        items.append(ExposureItem(ExposureCategory.COMPANY, e, reason, CrisisImpactDirection.UNCERTAIN, ImpactMagnitude.UNKNOWN))
    for a in cluster.affected_symbols:
        items.append(ExposureItem(ExposureCategory.ASSET_CLASS, a, "Named in reporting; not a trade recommendation.", CrisisImpactDirection.UNCERTAIN, ImpactMagnitude.UNKNOWN))
    for c in commodities:
        items.append(ExposureItem(ExposureCategory.COMMODITY, c, reason, CrisisImpactDirection.NEGATIVE, ImpactMagnitude.MODERATE))
    for c in currencies:
        items.append(ExposureItem(ExposureCategory.CURRENCY, c, reason, CrisisImpactDirection.UNCERTAIN, ImpactMagnitude.UNKNOWN))
    if event_type == CrisisEventType.ENERGY_DISRUPTION and not commodities:
        items.append(ExposureItem(ExposureCategory.COMMODITY, "crude oil", "Energy-disruption classification implies energy commodity exposure.", CrisisImpactDirection.NEGATIVE, ImpactMagnitude.HIGH))
        items.append(ExposureItem(ExposureCategory.SECTOR, "Energy", reason, CrisisImpactDirection.MIXED, ImpactMagnitude.HIGH))
    if event_type == CrisisEventType.SUPPLY_CHAIN_DISRUPTION and "semiconductor" in text.lower():
        items.append(ExposureItem(ExposureCategory.INDUSTRY, "Semiconductor manufacturing", "Supply-chain disruption involving critical manufacturing inputs.", CrisisImpactDirection.NEGATIVE, ImpactMagnitude.HIGH))
    return items


def compute_confidence(
    sources: Sequence[Source],
    article_count: int,
    has_conflicts: bool,
    is_stale: bool,
    event_type: CrisisEventType,
) -> float:
    if article_count == 0:
        return 0.0
    scores = [s.reliability_score for s in sources] or [0.5]
    base = sum(scores) / len(scores)
    independents = len({s.publisher for s in sources})
    if independents >= 2:
        base = min(1.0, base + 0.08)
    if any(s.is_primary for s in sources):
        base = min(1.0, base + 0.07)
    if has_conflicts:
        base = max(0.15, base - 0.25)
    if is_stale:
        base = max(0.1, base - 0.15)
    if event_type == CrisisEventType.OTHER:
        base = min(base, 0.45)
    if independents == 1 and not any(s.is_primary for s in sources):
        base = min(base, 0.65)
    return round(base, 3)


def probability_from_confidence(confidence: float, has_conflicts: bool) -> ProbabilityLevel:
    if has_conflicts or confidence < 0.4:
        return ProbabilityLevel.UNCERTAIN
    if confidence >= 0.8:
        return ProbabilityLevel.HIGH
    if confidence >= 0.55:
        return ProbabilityLevel.MODERATE
    return ProbabilityLevel.LOW


def risk_from_severity(sev: SeverityLevel) -> RiskLevel:
    return RiskLevel[sev.value] if sev.value in RiskLevel.__members__ else RiskLevel.UNKNOWN


def build_risk_dimensions(event_type: CrisisEventType, secondary: List[CrisisEventType], severity: SeverityLevel) -> RiskDimensions:
    types = {event_type, *secondary}
    sev_risk = risk_from_severity(severity)
    dims = RiskDimensions()

    def apply(attr: str, types_match: set):
        if types & types_match:
            setattr(dims, attr, sev_risk)

    apply("geopolitical_risk", {CrisisEventType.WAR, CrisisEventType.MILITARY_CONFLICT, CrisisEventType.TERRORISM_EVENT, CrisisEventType.DIPLOMATIC_CRISIS})
    apply("political_risk", {CrisisEventType.ELECTION, CrisisEventType.GOVERNMENT_CHANGE, CrisisEventType.POLITICAL_INSTABILITY})
    apply("trade_risk", {CrisisEventType.SANCTIONS, CrisisEventType.TARIFF, CrisisEventType.TRADE_RESTRICTION, CrisisEventType.EXPORT_CONTROL})
    apply("energy_risk", {CrisisEventType.ENERGY_DISRUPTION, CrisisEventType.WAR})
    apply("supply_chain_risk", {CrisisEventType.SUPPLY_CHAIN_DISRUPTION, CrisisEventType.EXPORT_CONTROL, CrisisEventType.NATURAL_DISASTER})
    apply("macroeconomic_risk", {CrisisEventType.INFLATION, CrisisEventType.CENTRAL_BANK, CrisisEventType.INTEREST_RATE, CrisisEventType.RECESSION_RISK, CrisisEventType.DEBT_CRISIS, CrisisEventType.CURRENCY_CRISIS})
    apply("regulatory_risk", {CrisisEventType.REGULATION})
    declared = [
        dims.geopolitical_risk, dims.political_risk, dims.trade_risk, dims.energy_risk,
        dims.supply_chain_risk, dims.macroeconomic_risk, dims.regulatory_risk,
    ]
    ranked = max(declared, key=lambda x: RISK_RANK[x])
    dims.overall_risk_level = ranked
    dims.overall_method = "MAX_OF_DECLARED_DIMENSIONS"
    return dims


def impact_from_type(event_type: CrisisEventType) -> Tuple[CrisisImpactDirection, ImpactMagnitude]:
    if event_type == CrisisEventType.OTHER:
        return CrisisImpactDirection.NO_MATERIAL_IMPACT_IDENTIFIED, ImpactMagnitude.UNKNOWN
    if event_type in (CrisisEventType.ELECTION, CrisisEventType.REGULATION):
        return CrisisImpactDirection.MIXED, ImpactMagnitude.LOW
    return CrisisImpactDirection.NEGATIVE, ImpactMagnitude.MODERATE


def build_timeline(cluster: NewsEventCluster) -> List[TimelineEntry]:
    articles = sorted(cluster.articles, key=lambda a: a.timestamp)
    entries: List[TimelineEntry] = []
    if not articles:
        return entries
    first = articles[0]
    src0 = cluster.sources[0].source_id if cluster.sources else "unknown"
    entries.append(TimelineEntry("Initial report", first.headline, first.timestamp, [src0]))
    for art, src in zip(articles[1:], cluster.sources[1:]):
        label = "Update"
        blob = (art.headline + " " + art.summary).lower()
        if any(k in blob for k in ESCALATE_KEYWORDS):
            label = "Escalation"
        elif any(k in blob for k in DEESCALATE_KEYWORDS + RESOLVE_KEYWORDS):
            label = "De-escalation or response"
        elif "government" in blob or "official" in blob:
            label = "Government response"
        entries.append(TimelineEntry(label, art.headline, art.timestamp, [src.source_id]))
    last = articles[-1]
    entries.append(TimelineEntry("Current state", last.headline, last.timestamp, [cluster.sources[-1].source_id if cluster.sources else src0]))
    return entries


def build_claims(cluster: NewsEventCluster) -> List[ClaimRecord]:
    claims = []
    for art, src in zip(cluster.articles, cluster.sources):
        kind = ClaimKind.CONFIRMED_FACT if src.is_primary else ClaimKind.REPORTED_CLAIM
        prefix = "Official source states" if src.is_primary else "Reporting claims"
        claims.append(ClaimRecord(
            statement=f"{prefix} ({src.publisher}): {art.headline}",
            kind=kind,
            source_ids=[src.source_id],
            confidence=src.reliability_score,
        ))
    return claims


def is_crisis_relevant(event_type: CrisisEventType) -> bool:
    return event_type != CrisisEventType.OTHER


class CrisisClassifier:
    """Maps a news cluster to deterministic crisis attributes."""

    def analyze_cluster(self, cluster: NewsEventCluster, now: Optional[datetime] = None):
        text = _corpus(cluster)
        primary, secondary = classify_event_types(text)
        countries = extract_countries(text, cluster)
        regions = extract_named(REGION_KEYWORDS, text)
        commodities = extract_named(COMMODITY_KEYWORDS, text)
        currencies = extract_named(CURRENCY_KEYWORDS, text)
        sources = cluster.sources
        has_primary = any(s.is_primary for s in sources)
        severity = classify_severity(primary, text, sources, len(cluster.articles))
        escalation = classify_escalation(text, len(cluster.articles), has_primary)
        scope = classify_scope(countries, text)
        horizon = classify_horizon(primary)
        confidence = compute_confidence(sources, len(cluster.articles), cluster.has_conflicts, cluster.is_stale, primary)
        transmission = build_transmission(primary, secondary, text)
        exposures = build_exposures(primary, countries, regions, cluster, commodities, currencies, text)
        direction, magnitude = impact_from_type(primary)
        if SEVERITY_RANK[severity] >= SEVERITY_RANK[SeverityLevel.HIGH]:
            magnitude = ImpactMagnitude.HIGH if magnitude != ImpactMagnitude.SEVERE else magnitude
        if severity in (SeverityLevel.SEVERE, SeverityLevel.CRITICAL):
            magnitude = ImpactMagnitude.SEVERE
        dims = build_risk_dimensions(primary, secondary, severity)
        risk = RiskFactors(
            severity=severity,
            probability=probability_from_confidence(confidence, cluster.has_conflicts),
            exposure=magnitude,
            horizon=horizon,
            dimensions=dims,
        )
        evidence = [
            Evidence(
                evidence_id=generate_id("ev"),
                description=a.headline,
                source_id=s.source_id,
                relevance_score=s.reliability_score,
            )
            for a, s in zip(cluster.articles, cluster.sources)
        ]
        uncertainty = []
        if cluster.has_conflicts:
            uncertainty.append("Sources disagree; claims are not silently reconciled.")
        if cluster.is_stale:
            uncertainty.append("Latest reporting may be stale.")
        if not has_primary:
            uncertainty.append("No primary official source in this cluster.")
        if horizon == CrisisTimeHorizon.UNKNOWN:
            uncertainty.append("Duration of the situation is not known.")
        if not countries:
            uncertainty.append("Geographic attribution is incomplete.")
        return {
            "event_type": primary,
            "secondary_types": secondary,
            "countries": countries,
            "regions": regions,
            "commodities": commodities,
            "currencies": currencies,
            "severity": severity,
            "escalation": escalation,
            "scope": scope,
            "horizon": horizon,
            "confidence": confidence,
            "transmission": transmission,
            "exposures": exposures,
            "impact_direction": direction,
            "impact_magnitude": magnitude,
            "risk": risk,
            "timeline": build_timeline(cluster),
            "claims": build_claims(cluster),
            "evidence": evidence,
            "uncertainty": uncertainty,
            "text": text,
        }
