"""Deterministic financial health and valuation math. No LLM arithmetic.

Formulas (skip and record INSUFFICIENT_DATA when a denominator is missing or <= 0):
  revenue_growth      = (revenue - prior_revenue) / prior_revenue
  earnings_growth     = (net_income - prior_net_income) / abs(prior_net_income)
  gross_margin        = gross_profit / revenue
  operating_margin    = operating_income / revenue  (falls back to ebit / revenue)
  net_margin          = net_income / revenue
  debt_to_equity      = total_debt / total_equity
  debt_to_cash        = total_debt / cash
  free_cash_flow_growth = (free_cash_flow - prior_free_cash_flow) / abs(prior_free_cash_flow)
  interest_coverage   = ebit / interest_expense   (ebit falls back to operating_income)
  pe_historical       = market_price / eps_actual
  pe_forward          = market_price / eps_estimate   (only if a real estimate exists)
  ps                  = market_cap / revenue   (or price / sales_per_share)
  pb                  = market_price / book_value_per_share
  ev_ebitda           = enterprise_value / ebitda
  fcf_yield           = free_cash_flow / market_cap
"""

from typing import Optional, Tuple
from app.core.models.deep_research import (
    ComputedValuation, FinancialHealthMetrics, ValuationBasis, ValuationContextLabel,
)
from app.core.models.fundamental import EarningsData, FinancialStatements, ValuationMetrics


def _ratio(numer: Optional[float], denom: Optional[float], abs_denom: bool = False) -> Optional[float]:
    if numer is None or denom is None:
        return None
    d = abs(denom) if abs_denom else denom
    if d == 0:
        return None
    return numer / d


def _growth(current: Optional[float], prior: Optional[float], abs_prior: bool = False) -> Optional[float]:
    if current is None or prior is None:
        return None
    d = abs(prior) if abs_prior else prior
    if d == 0:
        return None
    return (current - prior) / d


class FundamentalMetricsCalculator:
    def calculate_health(self, stmt: Optional[FinancialStatements]) -> FinancialHealthMetrics:
        missing = []
        formulas = {
            "revenue_growth": "(revenue - prior_revenue) / prior_revenue",
            "earnings_growth": "(net_income - prior_net_income) / abs(prior_net_income)",
            "gross_margin": "gross_profit / revenue",
            "operating_margin": "operating_income_or_ebit / revenue",
            "net_margin": "net_income / revenue",
            "debt_to_equity": "total_debt / total_equity",
            "debt_to_cash": "total_debt / cash",
            "free_cash_flow_growth": "(fcf - prior_fcf) / abs(prior_fcf)",
            "interest_coverage": "ebit_or_operating_income / interest_expense",
        }
        if stmt is None:
            return FinancialHealthMetrics(
                formulas=formulas,
                missing_fields=["financial_statements"],
            )
        op = stmt.operating_income if stmt.operating_income is not None else stmt.ebit
        ebit = stmt.ebit if stmt.ebit is not None else stmt.operating_income

        def need(name: str, value) -> Optional[float]:
            if value is None:
                missing.append(name)
            return value

        rev = need("revenue", stmt.revenue)
        metrics = FinancialHealthMetrics(
            revenue_growth=_growth(stmt.revenue, stmt.prior_revenue),
            earnings_growth=_growth(stmt.net_income, stmt.prior_net_income, abs_prior=True),
            gross_margin=_ratio(stmt.gross_profit, rev),
            operating_margin=_ratio(op, rev),
            net_margin=_ratio(stmt.net_income, rev),
            debt_to_equity=_ratio(stmt.total_debt, stmt.total_equity),
            debt_to_cash=_ratio(stmt.total_debt, stmt.cash),
            free_cash_flow_growth=_growth(stmt.free_cash_flow, stmt.prior_free_cash_flow, abs_prior=True),
            interest_coverage=_ratio(ebit, stmt.interest_expense),
            formulas=formulas,
            missing_fields=missing,
        )
        for field_name, val in (
            ("prior_revenue", stmt.prior_revenue),
            ("prior_net_income", stmt.prior_net_income),
            ("gross_profit", stmt.gross_profit),
            ("operating_income_or_ebit", op),
            ("net_income", stmt.net_income),
            ("total_debt", stmt.total_debt),
            ("total_equity", stmt.total_equity),
            ("cash", stmt.cash),
            ("prior_free_cash_flow", stmt.prior_free_cash_flow),
            ("interest_expense", stmt.interest_expense),
        ):
            if val is None and field_name not in metrics.missing_fields:
                # Only note if the derived metric is None
                pass
        if metrics.revenue_growth is None:
            metrics.missing_fields.append("revenue_growth_inputs")
        if metrics.interest_coverage is None:
            metrics.missing_fields.append("interest_coverage_inputs")
        metrics.missing_fields = list(dict.fromkeys(metrics.missing_fields))
        return metrics

    def calculate_valuation(
        self,
        stmt: Optional[FinancialStatements],
        provider_vals: Optional[ValuationMetrics],
        earnings: Optional[EarningsData],
        market_price: Optional[float],
    ) -> ComputedValuation:
        formulas = {
            "pe_historical": "market_price / eps_actual",
            "pe_forward": "market_price / eps_estimate (only if a real estimate exists)",
            "ps": "market_cap / revenue OR price / sales_per_share",
            "pb": "market_price / book_value_per_share",
            "ev_ebitda": "enterprise_value / ebitda",
            "fcf_yield": "free_cash_flow / market_cap",
        }
        missing = []
        price = market_price
        if price is None and provider_vals is not None:
            price = provider_vals.market_price

        eps_actual = earnings.eps_actual if earnings else None
        eps_est = earnings.eps_estimate if earnings else None
        pe_hist = _ratio(price, eps_actual)
        if pe_hist is None and provider_vals is not None:
            pe_hist = provider_vals.pe_ratio
        pe_fwd = _ratio(price, eps_est) if eps_est is not None else None
        if pe_fwd is None and provider_vals is not None:
            pe_fwd = provider_vals.forward_pe

        market_cap = provider_vals.market_cap if provider_vals else None
        revenue = stmt.revenue if stmt else None
        ps = _ratio(market_cap, revenue)
        if ps is None and provider_vals is not None:
            ps = provider_vals.ps_ratio or _ratio(price, provider_vals.sales_per_share)
        pb = None
        if provider_vals is not None:
            pb = provider_vals.pb_ratio or _ratio(price, provider_vals.book_value_per_share)
        ev = provider_vals.enterprise_value if provider_vals else None
        ebitda = stmt.ebitda if stmt else None
        ev_ebitda = _ratio(ev, ebitda)
        if ev_ebitda is None and provider_vals is not None:
            ev_ebitda = provider_vals.ev_to_ebitda
        fcf = stmt.free_cash_flow if stmt else None
        fcf_yield = _ratio(fcf, market_cap)
        if fcf_yield is None and provider_vals is not None:
            fcf_yield = provider_vals.fcf_yield

        basis = {
            "pe_historical": ValuationBasis.HISTORICAL if pe_hist is not None else ValuationBasis.INSUFFICIENT_DATA,
            "pe_forward": ValuationBasis.FORWARD_ESTIMATED if pe_fwd is not None else ValuationBasis.INSUFFICIENT_DATA,
            "ps": ValuationBasis.HISTORICAL if ps is not None else ValuationBasis.INSUFFICIENT_DATA,
            "pb": ValuationBasis.HISTORICAL if pb is not None else ValuationBasis.INSUFFICIENT_DATA,
            "ev_ebitda": ValuationBasis.HISTORICAL if ev_ebitda is not None else ValuationBasis.INSUFFICIENT_DATA,
            "fcf_yield": ValuationBasis.HISTORICAL if fcf_yield is not None else ValuationBasis.INSUFFICIENT_DATA,
        }
        if pe_hist is None:
            missing.append("pe_historical")
        if pe_fwd is None:
            missing.append("pe_forward")

        context = self._valuation_context(pe_hist, provider_vals)
        return ComputedValuation(
            pe_historical=pe_hist,
            pe_forward=pe_fwd,
            ps=ps,
            pb=pb,
            ev_ebitda=ev_ebitda,
            fcf_yield=fcf_yield,
            basis_notes=basis,
            context_labels=context,
            formulas=formulas,
            missing_fields=missing,
        )

    def _valuation_context(
        self, pe_hist: Optional[float], provider_vals: Optional[ValuationMetrics]
    ) -> list:
        labels = []
        if pe_hist is None:
            return [ValuationContextLabel.INSUFFICIENT_DATA]
        hist_ref = provider_vals.historical_pe if provider_vals else None
        peer = provider_vals.peer_pe if provider_vals else None
        if hist_ref is None and peer is None:
            return [ValuationContextLabel.INSUFFICIENT_DATA]
        if hist_ref is not None and hist_ref > 0:
            rel = pe_hist / hist_ref
            if rel < 0.9:
                labels.append(ValuationContextLabel.BELOW_OWN_HISTORY)
            elif rel > 1.1:
                labels.append(ValuationContextLabel.ABOVE_OWN_HISTORY)
            else:
                labels.append(ValuationContextLabel.IN_LINE_WITH_OWN_HISTORY)
        if peer is not None and peer > 0:
            rel = pe_hist / peer
            if rel < 0.9:
                labels.append(ValuationContextLabel.BELOW_PEER)
            elif rel > 1.1:
                labels.append(ValuationContextLabel.ABOVE_PEER)
            else:
                labels.append(ValuationContextLabel.IN_LINE_WITH_PEER)
        return labels or [ValuationContextLabel.INSUFFICIENT_DATA]
