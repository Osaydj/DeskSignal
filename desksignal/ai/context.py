"""Build a compact, deterministic evidence package for Claude."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from desksignal.analytics.executive import (
    monthly_performance,
    performance_by_dimension,
    unusual_monthly_changes,
)
from desksignal.analytics.kpis import calculate_kpis


UNUSUAL_CHANGE_THRESHOLD_PCT = 20.0
MAX_PRODUCT_EXTREMES = 5
MAX_DIMENSION_ITEMS = 20


def _round_money(value: float) -> float:
    return round(float(value), 2)


def _round_pct(value: float) -> float:
    return round(float(value), 2)


def _money(value: float) -> str:
    return f"${float(value):,.2f}"


def _percent(value: float) -> str:
    return f"{float(value):,.2f}%"


def _clean_label(value: object) -> str:
    """Keep uploaded labels readable and compact inside the AI request."""

    normalized = " ".join(str(value).replace("<", "").replace(">", "").split())
    return normalized[:120] or "Unknown"


def _optional_pct(value: object) -> float | None:
    return None if pd.isna(value) else _round_pct(float(value))


def _select_dimension_rows(
    performance: pd.DataFrame,
    *,
    extreme_size: int | None = None,
) -> pd.DataFrame:
    """Keep API payloads compact while retaining strongest and weakest rows."""

    if len(performance) <= MAX_DIMENSION_ITEMS:
        return performance.copy()

    size = extreme_size or MAX_DIMENSION_ITEMS // 2
    selected = pd.concat([performance.head(size), performance.tail(size)])
    return selected.drop_duplicates().reset_index(drop=True)


def _add_evidence(
    catalog: list[dict[str, str]],
    evidence_id: str,
    fact: str,
) -> str:
    catalog.append({"id": evidence_id, "fact": fact})
    return evidence_id


def _breakdown_payload(
    data: pd.DataFrame,
    dimension: str,
    catalog: list[dict[str, str]],
) -> dict[str, Any]:
    performance = performance_by_dimension(data, dimension)
    extreme_size = MAX_PRODUCT_EXTREMES if dimension == "product" else None
    selected = _select_dimension_rows(performance, extreme_size=extreme_size)
    records: list[dict[str, Any]] = []

    for position, (_, row) in enumerate(selected.iterrows(), start=1):
        label = _clean_label(row[dimension])
        prefix = f"{dimension}.{position:02d}"
        revenue_id = _add_evidence(
            catalog,
            f"{prefix}.revenue",
            f"{dimension.title()} {label} revenue: {_money(row['revenue'])}.",
        )
        profit_id = _add_evidence(
            catalog,
            f"{prefix}.profit",
            f"{dimension.title()} {label} profit: {_money(row['profit'])}.",
        )
        margin_id = _add_evidence(
            catalog,
            f"{prefix}.profit_margin",
            (
                f"{dimension.title()} {label} profit margin: "
                f"{_percent(row['profit_margin_pct'])}."
            ),
        )
        share_id = _add_evidence(
            catalog,
            f"{prefix}.revenue_share",
            (
                f"{dimension.title()} {label} share of filtered revenue: "
                f"{_percent(row['revenue_share_pct'])}."
            ),
        )

        records.append(
            {
                dimension: label,
                "revenue": _round_money(row["revenue"]),
                "profit": _round_money(row["profit"]),
                "profit_margin_pct": _round_pct(row["profit_margin_pct"]),
                "revenue_share_pct": _round_pct(row["revenue_share_pct"]),
                "units_sold": int(row["units_sold"]),
                "transactions": int(row["transactions"]),
                "evidence_ids": [revenue_id, profit_id, margin_id, share_id],
            }
        )

    return {
        "total_groups": int(len(performance)),
        "included_groups": int(len(selected)),
        "selection": (
            "all groups"
            if len(selected) == len(performance)
            else "highest and lowest revenue groups"
        ),
        "ranked_by": "revenue descending",
        "records": records,
    }


def build_analysis_context(data: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-ready calculated results and traceable evidence statements."""

    if data.empty:
        raise ValueError("AI analysis requires at least one sales row.")

    kpis = calculate_kpis(data)
    monthly = monthly_performance(data)
    changes = unusual_monthly_changes(
        monthly,
        threshold_pct=UNUSUAL_CHANGE_THRESHOLD_PCT,
    )
    catalog: list[dict[str, str]] = []

    scope_date_id = _add_evidence(
        catalog,
        "scope.date_range",
        (
            f"Filtered date range: {data['date'].min():%Y-%m-%d} through "
            f"{data['date'].max():%Y-%m-%d}."
        ),
    )
    transaction_id = _add_evidence(
        catalog,
        "kpi.total_transactions",
        f"Distinct transactions: {kpis.total_transactions:,}.",
    )
    revenue_id = _add_evidence(
        catalog,
        "kpi.total_revenue",
        f"Total revenue: {_money(kpis.total_revenue)}.",
    )
    profit_id = _add_evidence(
        catalog,
        "kpi.total_profit",
        f"Total profit: {_money(kpis.total_profit)}.",
    )
    margin_id = _add_evidence(
        catalog,
        "kpi.profit_margin",
        f"Overall profit margin: {_percent(kpis.profit_margin * 100)}.",
    )
    units_id = _add_evidence(
        catalog,
        "kpi.total_units_sold",
        f"Total units sold: {kpis.total_units_sold:,}.",
    )
    order_value_id = _add_evidence(
        catalog,
        "kpi.average_order_value",
        f"Average order value: {_money(kpis.average_order_value)}.",
    )
    best_product_id = _add_evidence(
        catalog,
        "kpi.best_product",
        (
            f"Highest-revenue product: {_clean_label(kpis.best_product)} at "
            f"{_money(kpis.best_product_revenue)}."
        ),
    )
    best_region_id = _add_evidence(
        catalog,
        "kpi.best_region",
        (
            f"Highest-revenue region: {_clean_label(kpis.best_region)} at "
            f"{_money(kpis.best_region_revenue)}."
        ),
    )

    monthly_records: list[dict[str, Any]] = []
    for _, row in monthly.iterrows():
        month = row["month"].strftime("%Y-%m")
        prefix = f"month.{month}"
        month_revenue_id = _add_evidence(
            catalog,
            f"{prefix}.revenue",
            f"Revenue in {month}: {_money(row['revenue'])}.",
        )
        month_profit_id = _add_evidence(
            catalog,
            f"{prefix}.profit",
            f"Profit in {month}: {_money(row['profit'])}.",
        )
        month_margin_id = _add_evidence(
            catalog,
            f"{prefix}.profit_margin",
            f"Profit margin in {month}: {_percent(row['profit_margin_pct'])}.",
        )
        month_evidence_ids = [
            month_revenue_id,
            month_profit_id,
            month_margin_id,
        ]
        revenue_change = _optional_pct(row["revenue_change_pct"])
        profit_change = _optional_pct(row["profit_change_pct"])
        if revenue_change is not None:
            month_evidence_ids.append(
                _add_evidence(
                    catalog,
                    f"{prefix}.revenue_change",
                    (
                        f"Revenue month-over-month change in {month}: "
                        f"{revenue_change:+,.2f}%."
                    ),
                )
            )
        if profit_change is not None:
            month_evidence_ids.append(
                _add_evidence(
                    catalog,
                    f"{prefix}.profit_change",
                    (
                        f"Profit month-over-month change in {month}: "
                        f"{profit_change:+,.2f}%."
                    ),
                )
            )
        monthly_records.append(
            {
                "month": month,
                "revenue": _round_money(row["revenue"]),
                "profit": _round_money(row["profit"]),
                "profit_margin_pct": _round_pct(row["profit_margin_pct"]),
                "transactions": int(row["transactions"]),
                "units_sold": int(row["units_sold"]),
                "revenue_change_pct": revenue_change,
                "profit_change_pct": profit_change,
                "evidence_ids": month_evidence_ids,
            }
        )

    unusual_records: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(changes.iterrows(), start=1):
        metric = str(row["metric"])
        from_month = row["from_month"].strftime("%Y-%m")
        to_month = row["to_month"].strftime("%Y-%m")
        change_pct = float(row["change_pct"])
        direction = "increased" if change_pct > 0 else "decreased"
        evidence_id = _add_evidence(
            catalog,
            f"unusual.{position:02d}.{metric}",
            (
                f"{metric.title()} {direction} by {_percent(abs(change_pct))} "
                f"from {_money(row['previous_value'])} in {from_month} to "
                f"{_money(row['current_value'])} in {to_month}; this met the "
                f"{_percent(UNUSUAL_CHANGE_THRESHOLD_PCT)} review threshold."
            ),
        )
        unusual_records.append(
            {
                "metric": metric,
                "from_month": from_month,
                "to_month": to_month,
                "previous_value": _round_money(row["previous_value"]),
                "current_value": _round_money(row["current_value"]),
                "change_pct": _round_pct(change_pct),
                "evidence_id": evidence_id,
            }
        )

    context: dict[str, Any] = {
        "data_scope": {
            "start_date": data["date"].min().strftime("%Y-%m-%d"),
            "end_date": data["date"].max().strftime("%Y-%m-%d"),
            "sales_rows": int(len(data)),
            "currency": "USD",
            "evidence_id": scope_date_id,
        },
        "kpis": {
            "total_revenue": _round_money(kpis.total_revenue),
            "total_profit": _round_money(kpis.total_profit),
            "profit_margin_pct": _round_pct(kpis.profit_margin * 100),
            "total_transactions": kpis.total_transactions,
            "total_units_sold": kpis.total_units_sold,
            "average_order_value": _round_money(kpis.average_order_value),
            "best_product": _clean_label(kpis.best_product),
            "best_product_revenue": _round_money(kpis.best_product_revenue),
            "best_region": _clean_label(kpis.best_region),
            "best_region_revenue": _round_money(kpis.best_region_revenue),
            "evidence_ids": [
                transaction_id,
                revenue_id,
                profit_id,
                margin_id,
                units_id,
                order_value_id,
                best_product_id,
                best_region_id,
            ],
        },
        "monthly_trends": monthly_records,
        "category_performance": _breakdown_payload(data, "category", catalog),
        "product_performance": _breakdown_payload(data, "product", catalog),
        "regional_performance": _breakdown_payload(data, "region", catalog),
        "unusual_changes": {
            "method": "absolute month-over-month change review rule",
            "threshold_pct": UNUSUAL_CHANGE_THRESHOLD_PCT,
            "records": unusual_records,
        },
        "calculation_notes": [
            "All monetary values are calculated by Python and expressed in USD.",
            "Profit margin equals profit divided by revenue.",
            "Best product and region are ranked by revenue.",
            "Unusual changes are transparent rule-based flags, not forecasts or statistical anomaly claims.",
        ],
        "evidence_catalog": catalog,
    }
    return context


def evidence_lookup(context: dict[str, Any]) -> dict[str, str]:
    """Return an evidence-ID-to-fact mapping and reject duplicate IDs."""

    lookup: dict[str, str] = {}
    for item in context.get("evidence_catalog", []):
        evidence_id = str(item["id"])
        if evidence_id in lookup:
            raise ValueError(f"Duplicate evidence ID: {evidence_id}")
        lookup[evidence_id] = str(item["fact"])
    return lookup


def context_fingerprint(context: dict[str, Any]) -> str:
    """Create a stable identifier used to prevent displaying stale summaries."""

    serialized = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
