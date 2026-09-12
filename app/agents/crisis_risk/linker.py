from typing import List, Optional
from app.agents.news_research.deduplicator import jaccard_similarity, tokenize_headline
from app.core.models.crisis import CrisisEventType, CrisisLink, GeopoliticalCrisis, SEVERITY_RANK, EscalationStatus


RELATED_TYPES = {
    (CrisisEventType.WAR, CrisisEventType.ENERGY_DISRUPTION),
    (CrisisEventType.MILITARY_CONFLICT, CrisisEventType.ENERGY_DISRUPTION),
    (CrisisEventType.SANCTIONS, CrisisEventType.TRADE_RESTRICTION),
    (CrisisEventType.TARIFF, CrisisEventType.SUPPLY_CHAIN_DISRUPTION),
    (CrisisEventType.EXPORT_CONTROL, CrisisEventType.SUPPLY_CHAIN_DISRUPTION),
    (CrisisEventType.ENERGY_DISRUPTION, CrisisEventType.INFLATION),
    (CrisisEventType.INFLATION, CrisisEventType.CENTRAL_BANK),
    (CrisisEventType.WAR, CrisisEventType.SANCTIONS),
}


def _types_related(a: CrisisEventType, b: CrisisEventType) -> bool:
    return a == b or (a, b) in RELATED_TYPES or (b, a) in RELATED_TYPES


class CrisisLinker:
    """Structural links between crises. Links are hypotheses, not proven causation."""

    def link(self, crises: List[GeopoliticalCrisis]) -> List[CrisisLink]:
        links: List[CrisisLink] = []
        for i, left in enumerate(crises):
            for right in crises[i + 1:]:
                overlap = set(left.countries_involved) & set(right.countries_involved)
                related = _types_related(left.event_type, right.event_type)
                title_sim = jaccard_similarity(tokenize_headline(left.title), tokenize_headline(right.title))
                if not related and not overlap and title_sim < 0.2:
                    continue
                if related and (overlap or title_sim >= 0.15):
                    links.append(CrisisLink(
                        from_event_id=left.event_id,
                        to_event_id=right.event_id,
                        relationship="RELATED_TRANSMISSION",
                        rationale=(
                            f"{left.event_type.value} may transmit into {right.event_type.value}; "
                            "this is a structured hypothesis, not proven causation."
                        ),
                        confidence=0.45 if overlap else 0.35,
                    ))
        return links


class CrisisRegistry:
    """Detects updates to previously observed crises instead of spawning duplicates."""

    def __init__(self):
        self._events: List[GeopoliticalCrisis] = []

    def find_match(self, candidate: GeopoliticalCrisis) -> Optional[GeopoliticalCrisis]:
        cand_tokens = tokenize_headline(candidate.title + " " + candidate.summary)
        for existing in self._events:
            if existing.event_type != candidate.event_type and not (
                set(existing.countries_involved) & set(candidate.countries_involved)
            ):
                continue
            exist_tokens = tokenize_headline(existing.title + " " + existing.summary)
            sim = jaccard_similarity(cand_tokens, exist_tokens)
            country_overlap = set(existing.countries_involved) & set(candidate.countries_involved)
            if sim >= 0.28 or (country_overlap and existing.event_type == candidate.event_type):
                return existing
        return None

    def remember(self, crisis: GeopoliticalCrisis) -> None:
        self._events = [e for e in self._events if e.event_id != crisis.event_id]
        self._events.append(crisis)

    def apply_update(self, previous: GeopoliticalCrisis, incoming: GeopoliticalCrisis) -> GeopoliticalCrisis:
        from app.core.models.crisis import CrisisUpdate
        changed = []
        if incoming.severity != previous.severity:
            changed.append("severity")
        if incoming.escalation_status != previous.escalation_status:
            changed.append("escalation_status")
        incoming.event_id = previous.event_id
        incoming.update = CrisisUpdate(
            previous_event_id=previous.event_id,
            change_summary="; ".join(changed) if changed else "Additional reporting on existing event.",
            prior_severity=previous.severity,
            new_severity=incoming.severity,
            prior_escalation=previous.escalation_status,
            new_escalation=incoming.escalation_status,
            changed_fields=changed or ["reporting"],
        )
        if SEVERITY_RANK[incoming.severity] > SEVERITY_RANK[previous.severity]:
            incoming.escalation_status = EscalationStatus.ESCALATING
            incoming.update.new_escalation = EscalationStatus.ESCALATING
        self.remember(incoming)
        return incoming
